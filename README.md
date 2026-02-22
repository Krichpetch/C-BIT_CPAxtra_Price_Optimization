# Retail Dynamic Pricing Optimization

A machine-learning-powered pricing engine for retail grocery, built for the 14-day horizon pricing competition. The system maximizes gross profit margin by combining demand forecasting, inventory-aware block pricing, and consumer psychology principles — while structurally avoiding instability penalties.

---

## How It Works

The pipeline runs end-to-end in a single script and produces a versioned submission CSV. At a high level:

1. **Load & Prep** — 8 raw CSV sources are ingested and merged. Pack size is parsed per SKU to drive the pricing discrimination logic downstream.
2. **ML Demand Forecast** — A `HistGradientBoostingRegressor` is trained on historical sales and used to predict average daily demand for every store × SKU pair across the 14-day window.
3. **Risk Flags** — OOS (out-of-stock) and XEL (cross-subcategory scarcity) risks are calculated from the forecast. High-risk items unlock premium pricing allowances, capped by pack size.
4. **Block Optimizer** — The 14 days are split into `[3, 4, 3, 4]` day blocks. One price is committed per block, chosen via grid search. Prices only change between blocks if the margin gain mathematically exceeds the instability penalty.
5. **Output** — Results are merged to the submission template, fallback-validated, snapped to psychological price endings (`.00` / `.50` / `.90`), and saved with an auto-incremented version number.

---

## Repository Structure

```
├── Raws/                          # Raw input data (not committed)
│   ├── sales_history.csv
│   ├── price_cost.csv
│   ├── inventory.csv              # Includes lead_time_days and on_hand
│   ├── competitor_prices.csv
│   ├── store_master.csv
│   ├── sku_master.csv
│   ├── sample_submission.csv
│   └── calendar_weather.csv
│
├── Submissions/                   # Auto-generated output CSVs
│   └── submission_chunking_lt_v{N}_{timestamp}.csv
│
├── leadtime_block_optimization.py # Main pricing model (latest)
├── compare_CSV.py                 # Financial comparison tool
├── inventory_graphs.py            # Inventory health dashboard
├── price_vs_demand_graphs.py      # Elasticity curve visualizer
├── seasons_graphs.py              # Seasonality & payday dashboard
└── README.md
```

---

## Pricing Logic

### Pack Size Discrimination
Pricing caps differ based on product format, reflecting real consumer behavior:

| Pack Type | Max Markup | Rationale |
|-----------|-----------|-----------|
| Single unit | +25% | Convenience shoppers don't calculate per-unit cost |
| Bulk / multi-pack | +5% | Value shoppers always compare unit prices |

### Risk-Based Premiums
- **OOS Risk (+15%):** If 14-day predicted demand exceeds on-hand stock, the item is flagged high risk and unlocks a +15% premium allowance.
- **XEL Risk (+10%):** If many SKUs within the same subcategory face OOS risk in a store, category-wide scarcity adds up to +10% more.
- Both premiums are hard-capped by the pack size rule above.

### Block Architecture
```
Aug 14─16   Aug 17─20   Aug 21─23   Aug 24─27
[Block 1]   [Block 2]   [Block 3]   [Block 4]
 3 days      4 days      3 days      4 days
```
One price is locked per block. Prices can change at block boundaries — but only if the profit gain is greater than the **5 THB instability penalty**.

### Scoring Formula
```
score = (margin × actual_units_sold)
      − (stockout_units × price)
      − instability_penalty
```

### Lead Time Replenishment
A reorder is assumed to be placed on Day 1 (Aug 14). If `lead_time_days` puts the arrival within the 14-day window, stock is injected into the inventory counter at the correct block boundary before that block's grid search runs. This prevents artificially high scarcity pricing on fast-restock SKUs.

### Zero Markdown Rule
The price floor is always `1.00 × regular_price`. The optimizer **cannot produce a discount** under any circumstance.

### Psychological Price Endings
All prices are snapped to `.00`, `.50`, or `.90` endings before output.

---

## Demand Model

**Algorithm:** `HistGradientBoostingRegressor` (scikit-learn)  
**Event boost:** Payday and holiday days get a `+20%` demand multiplier within the block scoring simulation.

**Features used:**

| Feature | Type | Description |
|--------|------|-------------|
| `store_id`, `sku_id` | Categorical | Store and product identity |
| `category`, `subcategory` | Categorical | Product hierarchy |
| `day_of_month`, `dow` | Numeric | Calendar position |
| `is_payday`, `is_holiday` | Binary | Event flags |
| `temp`, `rain_index` | Numeric | Weather effects |
| `price_paid` | Numeric | Own price signal |
| `comp_price` | Numeric | Competitor's last known price |
| `price_diff_vs_comp` | Numeric | Price gap vs. competitor |

### Per-SKU Price Elasticity

Rather than applying a single assumed elasticity to every product, the model estimates elasticity **per SKU from historical sales** using log-log OLS regression:

```
log(qty) = α + elasticity × log(price) + ε
```

The coefficient on `log(price)` is that SKU's price elasticity — how sensitive its demand is to price changes. A SKU with elasticity `-0.5` barely reacts to price moves (think staples like rice or eggs), while `-2.0` means demand drops sharply (think premium snacks or discretionary items).

**Estimation rules:**
- Requires at least **30 historical observations** per SKU
- Requires meaningful **price variation** (`std > 0.01`) in the history
- Estimated elasticity must fall within **[-5.0, -0.1]** — values outside this range are almost certainly noise or data artifacts
- SKUs that fail any of these checks fall back to **-1.3** (a standard grocery assumption)

The summary report prints how many SKUs were estimated from data vs. how many used the fallback, and the average estimated elasticity across the assortment.

---

## Utility Scripts

### `compare_CSV.py`
Compares two submission CSVs and simulates real-world financial impact. Outputs a `.txt` report with:
- Row-level price change summary (how many prices changed, direction, average delta)
- 14-day gross profit simulation for both files using elasticity-adjusted demand
- Clear winner declaration with THB profit difference and % improvement

**Usage:**
```python
compare_csv_files(
    file1_path='Submissions/submission_baseline.csv',
    file2_path='Submissions/submission_chunking_lt_v1_....csv',
    output_file='comparison_report.txt',
    run_financials=True
)
```

### `inventory_graphs.py`
Generates the **Inventory Health & Stockout Risk Dashboard** (4-panel figure):
- Distribution of days of stock remaining across all store-SKU pairs
- 14-day demand vs. on-hand scatter (stockout danger zone highlighted)
- Count of items by risk level (High / Medium / Low)
- Top 10 worst unit deficits by store and SKU

Requires `stockout_risk_report_ultimate.csv` in `Raws/`.

### `price_vs_demand_graphs.py`
Generates the **ML Elasticity Curve** — a bar + line chart showing how the `-1.3` elasticity constant translates pricing decisions (−30% to +30%) into predicted daily demand. Useful for communicating the model's core mechanics to stakeholders.

### `seasons_graphs.py`
Generates the **Seasonality & Sales Opportunity Dashboard** (4-panel figure):
- Average sales volume by day of week
- Payday spike effect with % lift annotation
- Predicted demand strength across the Aug 14–27 window
- Traffic intensity action plan (high / normal / low traffic days)

---

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib
```

Place all raw CSV files into a `Raws/` folder in the same directory as the scripts. Then run:

```bash
python leadtime_block_optimization.py
```

Output will be saved to `Submissions/submission_chunking_lt_v{N}_{timestamp}.csv`.

---

## Output Format

| Column | Description |
|--------|-------------|
| `ID` | Row identifier from submission template |
| `store_id` | Store identifier |
| `sku_id` | Product identifier |
| `date` | Pricing date (Aug 14–27) |
| `proposed_price` | Final recommended price in THB |
