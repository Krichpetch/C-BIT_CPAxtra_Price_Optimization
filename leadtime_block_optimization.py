"""
Retail Pricing Optimization - SEQUENTIAL CHUNKING MODEL (WITH LEAD TIME REPLENISHMENT)
=======================================================================================
Goal: Enable dynamic pricing to capture event volume and protect inventory,
      while strictly minimizing Kaggle instability penalties.

LOGIC LAYERS:
1. Sequential Chunking: The 14-day horizon is split into [3, 4, 3, 4] day blocks.
   Prices can change between blocks, guaranteeing a minimum 3-day hold.
2. Dynamic Depletion + Replenishment: Inventory drains block-by-block. If a SKU's
   lead_time_days means a reorder arrives within the 14-day window, stock is
   replenished at the correct block boundary. Prevents over-pricing fast-restock items.
3. Penalty Mathematics: Strictly subtracts the -5 point Kaggle Instability
   Penalty during Grid Search. Only changes price if margin gained > penalty.
4. Pack Size Discrimination: Singles capped at +25%, Bulk capped at +5%.
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
import re
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 0. PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'Raws')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'Submissions')

def raw(filename):
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path) and os.path.exists(filename):
        return filename
    return path

def next_submission_path():
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(SUBMISSIONS_DIR, 'submission_chunking_lt_v*.csv'))
    versions = []
    for f in existing:
        base = os.path.basename(f)
        try:
            v = int(base.split('_v')[1].split('_')[0])
            versions.append(v)
        except (IndexError, ValueError):
            pass
    next_v   = max(versions, default=0) + 1
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'submission_chunking_lt_v{next_v}_{ts}.csv'
    return os.path.join(SUBMISSIONS_DIR, filename)

# ─────────────────────────────────────────────
# 1. LOAD DATA & PACK SIZE PARSING
# ─────────────────────────────────────────────
print("Loading data...")
sales  = pd.read_csv(raw('sales_history.csv'), parse_dates=['date'])
pc     = pd.read_csv(raw('price_cost.csv'))
inv    = pd.read_csv(raw('inventory.csv'))
comp   = pd.read_csv(raw('competitor_prices.csv'), parse_dates=['date'])
store  = pd.read_csv(raw('store_master.csv'))
sku_m  = pd.read_csv(raw('sku_master.csv'))
subm   = pd.read_csv(raw('sample_submission.csv'))
cw     = pd.read_csv(raw('calendar_weather.csv'), parse_dates=['date'])

sku_subcat_dict = sku_m.set_index('sku_id')['subcategory'].to_dict()

def parse_pack_size(val):
    try:
        if pd.isna(val): return 1.0
        if isinstance(val, str):
            numbers = re.findall(r'\d+', val)
            return float(numbers[0]) if numbers else 1.0
        return float(val)
    except:
        return 1.0

PACK_COL = 'pack_size' if 'pack_size' in sku_m.columns else sku_m.columns[2]
sku_m['parsed_pack'] = sku_m[PACK_COL].apply(parse_pack_size)
sku_pack_dict = sku_m.set_index('sku_id')['parsed_pack'].to_dict()

# ─────────────────────────────────────────────
# 2. DYNAMIC ML FORECAST
# ─────────────────────────────────────────────
print("\n--- Phase 1: Training ML Demand Model ---")
train_df = sales.merge(cw, on='date', how='left')
train_df['day_of_month'] = train_df['date'].dt.day
train_df = train_df.merge(sku_m[['sku_id', 'category', 'subcategory']], on='sku_id', how='left')
train_df = train_df.merge(comp, on=['sku_id', 'date'], how='left')
train_df['comp_price'] = train_df['comp_price'].fillna(train_df['price_paid'])
train_df['price_diff_vs_comp'] = train_df['comp_price'] - train_df['price_paid']

cat_cols = ['store_id', 'sku_id', 'category', 'subcategory']
cat_mappings = {}
for col in cat_cols:
    train_df[col] = train_df[col].astype('category')
    cat_mappings[col] = dict(enumerate(train_df[col].cat.categories))
    train_df[f'{col}_idx'] = train_df[col].cat.codes

features = [
    'store_id_idx', 'sku_id_idx', 'category_idx', 'subcategory_idx',
    'day_of_month', 'dow', 'is_payday', 'is_holiday',
    'temp', 'rain_index', 'price_paid', 'comp_price', 'price_diff_vs_comp'
]

model = HistGradientBoostingRegressor(
    categorical_features=[0, 1, 2, 3], random_state=42, max_iter=100
)
model.fit(train_df[features], train_df['qty'])

# ─────────────────────────────────────────────
# ELASTICITY ESTIMATION (Per-SKU, Data-Derived)
# ─────────────────────────────────────────────
# Instead of assuming a fixed -1.3 for every SKU, we estimate each SKU's
# price elasticity directly from historical sales using log-log regression:
#   log(qty) = α + elasticity × log(price) + ε
# The coefficient on log(price) IS the elasticity for that SKU.
# SKUs with too few observations or no price variation fall back to -1.3.

print("--- Phase 1b: Estimating Per-SKU Price Elasticity from History ---")

FALLBACK_ELASTICITY = -1.3   # Sensible grocery default if data is insufficient
MIN_OBS             = 30     # Minimum rows needed to trust the regression
MIN_PRICE_STD       = 0.01   # Minimum price variation (avoids division near zero)

sku_elasticity_dict = {}

for sku_id, group in train_df.groupby('sku_id'):
    # Need enough rows and actual price variation to fit a meaningful regression
    if len(group) < MIN_OBS or group['price_paid'].std() < MIN_PRICE_STD:
        sku_elasticity_dict[sku_id] = FALLBACK_ELASTICITY
        continue

    # Drop rows where qty or price are zero/negative (can't take log)
    g = group[(group['qty'] > 0) & (group['price_paid'] > 0)].copy()
    if len(g) < MIN_OBS:
        sku_elasticity_dict[sku_id] = FALLBACK_ELASTICITY
        continue

    log_price = np.log(g['price_paid']).values.reshape(-1, 1)
    log_qty   = np.log(g['qty']).values

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(log_price, log_qty)
    estimated = reg.coef_[0]

    # Sanity bounds: elasticity should be negative and not wildly extreme
    # Values outside [-5, -0.1] are almost certainly noise or data artifacts
    if -5.0 <= estimated <= -0.1:
        sku_elasticity_dict[sku_id] = estimated
    else:
        sku_elasticity_dict[sku_id] = FALLBACK_ELASTICITY

n_estimated  = sum(1 for v in sku_elasticity_dict.values() if v != FALLBACK_ELASTICITY)
n_fallback   = len(sku_elasticity_dict) - n_estimated
avg_estimated = np.mean([v for v in sku_elasticity_dict.values() if v != FALLBACK_ELASTICITY]) if n_estimated else FALLBACK_ELASTICITY

print(f"  Elasticity estimated from data : {n_estimated} SKUs  (avg: {avg_estimated:.3f})")
print(f"  Fallback to {FALLBACK_ELASTICITY}            : {n_fallback} SKUs")

# ─────────────────────────────────────────────
# PAYDAY LIFT ESTIMATION (Data-Derived)
# ─────────────────────────────────────────────
# Instead of hardcoding +20%, calculate the real payday demand multiplier
# from historical sales by comparing avg qty on payday vs. normal days.
# Falls back to 1.20 if the calendar data doesn't have enough payday rows.

print("--- Phase 1c: Estimating Payday Demand Lift from History ---")

FALLBACK_PAYDAY_LIFT   = 1.20   # Used if data is insufficient
MIN_PAYDAY_OBS         = 10     # Minimum payday days in history to trust the estimate

if 'is_payday' in train_df.columns:
    payday_avg = train_df[train_df['is_payday'] == 1]['qty'].mean()
    normal_avg = train_df[train_df['is_payday'] == 0]['qty'].mean()
    payday_obs = train_df['is_payday'].sum()

    if payday_obs >= MIN_PAYDAY_OBS and normal_avg > 0:
        PAYDAY_LIFT = payday_avg / normal_avg
        # Sanity bound: lift should be positive and not absurdly large
        if not (1.0 <= PAYDAY_LIFT <= 3.0):
            PAYDAY_LIFT = FALLBACK_PAYDAY_LIFT
    else:
        PAYDAY_LIFT = FALLBACK_PAYDAY_LIFT
else:
    PAYDAY_LIFT = FALLBACK_PAYDAY_LIFT

print(f"  Payday demand lift : {PAYDAY_LIFT:.3f}x  ({(PAYDAY_LIFT - 1)*100:+.1f}% vs normal days)")

# ─────────────────────────────────────────────
# FALLBACK DEMAND (Data-Derived)
# ─────────────────────────────────────────────
# When a store-SKU pair has no forecast (new listing, missing data, etc.),
# fall back to the store's own average daily sales rather than a fixed 5.0.
# If the store itself has no history, use the global average.

GLOBAL_AVG_DAILY_DEMAND = train_df.groupby(['store_id', 'sku_id'])['qty'].mean().mean()
store_avg_daily_demand  = train_df.groupby('store_id')['qty'].mean().to_dict()

print(f"  Global avg daily demand fallback : {GLOBAL_AVG_DAILY_DEMAND:.2f} units")

# ─────────────────────────────────────────────
# INSTABILITY PENALTY
# ─────────────────────────────────────────────
# Named constant for the THB cost of changing price between blocks.
# Set to 5.0 to match the Kaggle competition rule.
# In a real retail deployment, set this to 0.0 or your actual repricing cost.

INSTABILITY_PENALTY_THB = 5.0

print("--- Phase 2: Predicting Base Demand & Ecosystem Risk ---")
future_dates = pd.date_range('2025-08-14', '2025-08-27')
future_cw = cw[cw['date'].isin(future_dates)]

last_comp_prices = (
    comp.sort_values('date')
    .groupby('sku_id').tail(1)
    .set_index('sku_id')['comp_price']
    .to_dict()
)
reg_prices = pc.set_index('sku_id')['regular_price'].to_dict()

# ── Pull lead_time_days alongside on_hand ──
store_skus = inv[['store_id', 'sku_id', 'lead_time_days', 'on_hand']].drop_duplicates()
store_skus = store_skus.merge(
    sku_m[['sku_id', 'category', 'subcategory']], on='sku_id', how='left'
)

future_rows = []
for d in future_dates:
    cw_row = future_cw[future_cw['date'] == d].iloc[0]
    temp_df = store_skus.copy()
    temp_df['date'] = d
    temp_df['day_of_month'] = d.day
    temp_df['dow'] = cw_row['dow']
    temp_df['is_payday'] = cw_row['is_payday']
    temp_df['is_holiday'] = cw_row['is_holiday']
    temp_df['temp'] = cw_row['temp']
    temp_df['rain_index'] = cw_row['rain_index']
    temp_df['price_paid'] = temp_df['sku_id'].map(reg_prices)
    temp_df['comp_price'] = temp_df['sku_id'].map(last_comp_prices).fillna(temp_df['price_paid'])
    temp_df['price_diff_vs_comp'] = temp_df['comp_price'] - temp_df['price_paid']
    future_rows.append(temp_df)

future_df = pd.concat(future_rows, ignore_index=True)

for col in cat_cols:
    mapping_dict = {v: k for k, v in cat_mappings[col].items()}
    future_df[f'{col}_idx'] = future_df[col].map(mapping_dict).fillna(-1).astype(int)

future_df['pred_qty'] = np.maximum(0, model.predict(future_df[features]))

projected_demand = (
    future_df.groupby(['store_id', 'sku_id'])['pred_qty']
    .mean().reset_index()
)
base_demand_dict = projected_demand.set_index(['store_id', 'sku_id'])['pred_qty'].to_dict()

inv_dict = {
    (r['store_id'], r['sku_id']): r['on_hand']
    for _, r in inv.iterrows()
}

# ── Lead time lookup: (store_id, sku_id) → lead_time_days ──
lead_time_dict = {
    (r['store_id'], r['sku_id']): int(r['lead_time_days'])
    for _, r in inv.iterrows()
}

# Risk flags
risk_analysis = []
for (store_id, sku_id), base_dem in base_demand_dict.items():
    on_hand  = inv_dict.get((store_id, sku_id), 0)
    subcat   = sku_subcat_dict.get(sku_id, 'Unknown')
    is_high_risk = 1 if (base_dem * 14) > on_hand else 0
    risk_analysis.append({
        'store_id': store_id, 'sku_id': sku_id,
        'subcategory': subcat, 'is_high_risk': is_high_risk
    })

risk_df = pd.DataFrame(risk_analysis)
subcat_risk      = risk_df.groupby(['store_id', 'subcategory'])['is_high_risk'].mean().to_dict()
is_high_risk_dict = risk_df.set_index(['store_id', 'sku_id'])['is_high_risk'].to_dict()

event_days = {
    d: bool(
        future_cw[future_cw['date'] == d].iloc[0]['is_payday'] == 1 or
        future_cw[future_cw['date'] == d].iloc[0]['is_holiday'] == 1
    )
    for d in future_dates
}

# ─────────────────────────────────────────────
# 3. BLOCK DEFINITIONS & LEAD TIME HELPERS
# ─────────────────────────────────────────────

HORIZON_START = pd.Timestamp('2025-08-14')

# [3, 4, 3, 4] day blocks
blocks = [
    pd.date_range('2025-08-14', periods=3).tolist(),   # Block 1
    pd.date_range('2025-08-17', periods=4).tolist(),   # Block 2
    pd.date_range('2025-08-21', periods=3).tolist(),   # Block 3
    pd.date_range('2025-08-24', periods=4).tolist(),   # Block 4
]

# First date of each block — used to check if a reorder has arrived
block_start_dates = [b[0] for b in blocks]


def get_reorder_qty(base_dem, block_days):
    """
    Conservative reorder quantity: enough to cover one block's expected demand.
    You can replace this with a fixed MOQ or a field from your data if available.
    """
    return base_dem * block_days


def replenishment_for_block(block_idx, lead_time_days, base_dem):
    """
    Returns the quantity of stock that arrives AT the start of `block_idx`,
    based on a reorder triggered on Day 1 of the horizon (Aug 14).

    Logic:
      - We assume a reorder is placed on the horizon start date.
      - Arrival date = HORIZON_START + lead_time_days.
      - Find which block that arrival date falls into.
      - Inject stock at the START of that block (and only once).

    Returns: replenishment qty (float) if this is the arrival block, else 0.
    """
    arrival_date = HORIZON_START + timedelta(days=lead_time_days)

    # Which block does the arrival fall into?
    arrival_block = None
    for i, block in enumerate(blocks):
        if block[0] <= arrival_date <= block[-1]:
            arrival_block = i
            break
    else:
        # Arrival is beyond the 14-day window — no replenishment this horizon
        return 0.0

    if block_idx == arrival_block:
        # Stock arrives at the start of this block
        reorder_qty = get_reorder_qty(base_dem, len(blocks[arrival_block]))
        return reorder_qty

    return 0.0


# ─────────────────────────────────────────────
# 4. ALLOWED PRICE ENDINGS & ROUNDING
# ─────────────────────────────────────────────
ALLOWED_ENDINGS = [0.00, 0.50, 0.90]

def snap_to_allowed(price):
    floor_int = int(price)
    candidates = [
        round(b + e, 2)
        for b in [floor_int - 1, floor_int, floor_int + 1]
        for e in ALLOWED_ENDINGS
        if b + e > 0
    ]
    return sorted(candidates, key=lambda x: abs(x - price))[0]

def first_valid_ending_above(floor):
    for base in range(int(floor), int(floor) + 3):
        for end in ALLOWED_ENDINGS:
            candidate = round(base + end, 2)
            if candidate >= floor:
                return candidate
    return snap_to_allowed(floor)

def last_valid_ending_below(ceiling):
    for base in range(int(ceiling) + 1, int(ceiling) - 2, -1):
        for end in sorted(ALLOWED_ENDINGS, reverse=True):
            candidate = round(base + end, 2)
            if candidate <= ceiling:
                return candidate
    return snap_to_allowed(ceiling)

def build_price_grid(min_p, max_p):
    lo = max(0, int(min_p) - 1)
    hi = int(max_p) + 2
    grid = [
        round(b + e, 2)
        for b in range(lo, hi + 1)
        for e in ALLOWED_ENDINGS
        if min_p <= round(b + e, 2) <= max_p
    ]
    return sorted(set(grid))

# ─────────────────────────────────────────────
# 5. PROFIT OPTIMIZATION (SEQUENTIAL CHUNKING + LEAD TIME)
# ─────────────────────────────────────────────
print("\n--- Phase 3: Optimizing Sequential Blocks (Min 3-Day Holds + Lead Time) ---")

pc_dict = pc.set_index('sku_id').to_dict('index')
results = []

for (store_id, sku_id), _ in sales.groupby(['store_id', 'sku_id']):
    sku_data = pc_dict.get(sku_id)
    if sku_data is None:
        continue

    regular_price = sku_data['regular_price']
    unit_cost     = sku_data['unit_cost']
    vat_rate      = sku_data['vat_rate']

    key    = (store_id, sku_id)
    subcat = sku_subcat_dict.get(sku_id, 'Unknown')

    my_risk        = is_high_risk_dict.get(key, 0)
    my_subcat_risk = subcat_risk.get((store_id, subcat), 0.0)

    # ── PACK SIZE DISCRIMINATION ──
    pack_size   = sku_pack_dict.get(sku_id, 1.0)
    sku_max_cap = 0.05 if pack_size > 1.0 else 0.25

    # ── PRICE BOUNDS ──
    requested_premium = 0.0
    if my_risk == 1:
        requested_premium += 0.15
    requested_premium += (0.10 * my_subcat_risk)
    final_premium_cap = min(requested_premium, sku_max_cap)

    markdown_floor = regular_price * 1.00   # NO MARKDOWNS
    price_ceiling  = regular_price * (1.00 + final_premium_cap)
    cost_floor     = unit_cost * (1 + vat_rate)

    effective_min = max(cost_floor, markdown_floor)
    effective_max = price_ceiling
    if effective_min > effective_max:
        effective_max = effective_min

    price_grid = build_price_grid(effective_min, effective_max)
    if not price_grid:
        price_grid = [first_valid_ending_above(effective_min)]

    # Use data-derived fallback: store average → global average (never a magic number)
    _store_fallback = store_avg_daily_demand.get(store_id, GLOBAL_AVG_DAILY_DEMAND)
    base_dem   = base_demand_dict.get(key, _store_fallback)
    # Use data-derived elasticity for this SKU, fall back to -1.3 if unavailable
    elasticity = sku_elasticity_dict.get(sku_id, FALLBACK_ELASTICITY)

    # ── LEAD TIME: how many days until restock arrives ──
    lead_time_days = lead_time_dict.get(key, 999)   # 999 = no restock this horizon

    # Starting inventory and previous block price
    current_inv = inv_dict.get(key, 0)
    prev_price  = regular_price

    # ── EVALUATE EACH BLOCK SEQUENTIALLY ──
    for block_idx, block_dates in enumerate(blocks):

        # ── REPLENISHMENT INJECTION ──
        # Check if a reorder placed on Day 1 arrives at the start of this block.
        restock_qty = replenishment_for_block(block_idx, lead_time_days, base_dem)
        if restock_qty > 0:
            current_inv += restock_qty
            print(f"  [RESTOCK] store={store_id} sku={sku_id} "
                  f"block={block_idx+1} +{restock_qty:.1f} units "
                  f"(lead_time={lead_time_days}d)")

        best_price = price_grid[-1]
        best_score = -np.inf
        best_sales = 0.0

        for p in price_grid:
            # Sum demand across days in this block
            block_demand = 0.0
            for d in block_dates:
                daily_dem = base_dem * ((p / regular_price) ** elasticity)
                if event_days[d]:
                    daily_dem *= PAYDAY_LIFT   # Data-derived payday/holiday boost
                block_demand += daily_dem

            actual_units   = min(block_demand, current_inv)
            stockout_units = max(0.0, block_demand - current_inv)
            margin         = p - unit_cost

            # Instability penalty: only change price if gain justifies it
            instability_penalty = INSTABILITY_PENALTY_THB if p != prev_price else 0.0

            score = (margin * actual_units) - (stockout_units * p) - instability_penalty

            if score > best_score:
                best_score = score
                best_price = p
                best_sales = actual_units

        # Commit block price to results
        for d in block_dates:
            results.append({
                'store_id':       store_id,
                'sku_id':         sku_id,
                'date':           d,
                'proposed_price': best_price
            })

        # Drain inventory and carry state to next block
        current_inv = max(0.0, current_inv - best_sales)
        prev_price  = best_price

results_df = pd.DataFrame(results)

# ─────────────────────────────────────────────
# 6. BUILD FINAL SUBMISSION FILE & FALLBACKS
# ─────────────────────────────────────────────
print("\nEnforcing final constraints...")
subm['date_parsed'] = pd.to_datetime(subm['date'], dayfirst=True)
subm_merged = subm.merge(
    results_df,
    left_on  =['store_id', 'sku_id', 'date_parsed'],
    right_on =['store_id', 'sku_id', 'date'],
    how      ='left',
    suffixes =('_orig', '_new')
)

pc_reg = pc.set_index('sku_id')['regular_price'].to_dict()

def master_fallback(row):
    sku_id   = row['sku_id']
    base_reg = pc_reg.get(sku_id, 1.0)
    p        = row['proposed_price_new']
    pack_val = sku_pack_dict.get(sku_id, 1.0)
    max_cap  = 1.05 if pack_val > 1.0 else 1.25

    if pd.isna(p) or p < base_reg:
        return first_valid_ending_above(base_reg)
    elif p > (base_reg * max_cap):
        return last_valid_ending_below(base_reg * max_cap)
    return p

subm_merged['proposed_price'] = subm_merged.apply(master_fallback, axis=1)

final = subm_merged[['ID', 'store_id', 'sku_id', 'date_orig', 'proposed_price']].copy()
final.columns = ['ID', 'store_id', 'sku_id', 'date', 'proposed_price']

output_path = next_submission_path()
final.to_csv(output_path, index=False)

print(f"\n{'─'*60}")
print(f"  ✓ Lead-Time Chunking Submission saved: {output_path}")
print(f"{'─'*60}")

# ─────────────────────────────────────────────
# 7. SUMMARY REPORT
# ─────────────────────────────────────────────
pc_merged = final.merge(pc, on='sku_id')
pc_merged['discount_pct'] = (
    (pc_merged['regular_price'] - pc_merged['proposed_price'])
    / pc_merged['regular_price']
) * 100

print("\n=== SEQUENTIAL CHUNKING + LEAD TIME SUMMARY ===")
print(f"  Avg price vs regular   : {-pc_merged['discount_pct'].mean():+.2f}%")
print(f"  Items Priced Above Reg : {(pc_merged['discount_pct'] < -0.01).mean()*100:.1f}% of rows")
print(f"  Highest markup allowed : Singles (+25%), Bulk (+5%)")
print(f"\n  Elasticity (data-derived) : {n_estimated} SKUs (avg {avg_estimated:.3f})")
print(f"  Elasticity (fallback)     : {n_fallback} SKUs → {FALLBACK_ELASTICITY}")
print(f"  Payday lift               : {PAYDAY_LIFT:.3f}x ({(PAYDAY_LIFT-1)*100:+.1f}%) — data-derived")
print(f"  Demand fallback           : store avg → global avg ({GLOBAL_AVG_DAILY_DEMAND:.2f} units/day)")
print("\n[!] ARCHITECTURE NOTES:")
print("  - 14 days split into blocks [3, 4, 3, 4] days long.")
print("  - Inventory depletes block-by-block.")
print("  - Restock injected at the correct block if lead_time_days puts")
print("    arrival within the 14-day window. Prevents over-pricing fast-")
print("    restock SKUs in later blocks.")
print()