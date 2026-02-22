import pandas as pd
import numpy as np
import os

# =========================
# 1️⃣ Path Configuration
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'Raws')

def load_data(filename):
    """Helper to load data from Raws/ or current directory."""
    raw_path = os.path.join(RAW_DIR, filename)
    local_path = os.path.join(BASE_DIR, filename)
    
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    elif os.path.exists(local_path):
        return pd.read_csv(local_path)
    else:
        raise FileNotFoundError(f"❌ File not found: {filename}")

# =========================
# 2️⃣ Analysis Logic
# =========================

def analyze_store_performance():
    print("Loading data...")
    try:
        sales = load_data('sales_history.csv')
        stores = load_data('store_master.csv')
    except Exception as e:
        print(e)
        return

    # 1. Calculate Store Metrics from Sales History
    print("Calculating store metrics...")
    
    # Revenue = Qty * Price Paid
    sales['revenue'] = sales['qty'] * sales['price_paid']

    store_stats = sales.groupby('store_id').agg({
        'revenue': 'sum',
        'qty': 'sum',
        'price_paid': 'mean' # Avg price per unit sold
    }).reset_index()

    store_stats.columns = ['store_id', 'total_revenue', 'total_units', 'avg_ticket_price']

    # 2. Merge with Store Master to get Income Index & Type
    report = pd.merge(store_stats, stores, on='store_id', how='left')

    # 3. Define Pricing Strategy based on Income Index
    def recommend_strategy(index):
        if index >= 1.10:
            return "PREMIUM (+5%)", 1.05
        elif index >= 1.02:
            return "Standard (+0%)", 1.00
        elif index >= 0.90:
            return "Value (-2%)", 0.98
        else:
            return "DISCOUNT (-5%)", 0.95

    report['strategy_label'], report['price_multiplier'] = zip(*report['income_index'].apply(recommend_strategy))

    # 4. Sorting for the Report (Rank by Revenue)
    report = report.sort_values('total_revenue', ascending=False)

    # =========================
    # 3️⃣ Generate Reports
    # =========================

    print("\n" + "="*60)
    print(" 🏪 STORE PERFORMANCE & PRICING STRATEGY REPORT")
    print("="*60)

    print("\n📊 TOP 5 PERFORMING STORES (By Revenue)")
    print("-" * 60)
    # Format currency for readability
    top_cols = ['store_id', 'region', 'store_type', 'total_revenue', 'income_index']
    print(report[top_cols].head(5).to_string(index=False))

    print("\n📉 BOTTOM 5 STORES (Potential Risks)")
    print("-" * 60)
    print(report[top_cols].tail(5).to_string(index=False))

    print("\n💰 RECOMMENDED PRICING ADAPTATION (By Income Level)")
    print("   (Strategy: Charge premium in wealthy areas, discount in competitive ones)")
    print("-" * 80)
    
    # Show a sample of stores with different strategies
    strategy_cols = ['store_id', 'region', 'income_index', 'strategy_label', 'price_multiplier']
    
    # Pick examples from each tier
    high_income = report[report['income_index'] >= 1.05].head(2)
    mid_income  = report[(report['income_index'] > 0.95) & (report['income_index'] < 1.05)].head(2)
    low_income  = report[report['income_index'] <= 0.95].head(2)
    
    display_sample = pd.concat([high_income, mid_income, low_income])
    print(display_sample[strategy_cols].to_string(index=False))

    # =========================
    # 4️⃣ Save to CSV
    # =========================
    output_file = "store_pricing_strategy.csv"
    report.to_csv(output_file, index=False)
    print(f"\n✅ Full analysis saved to: {output_file}")
    print("   Use 'price_multiplier' from this file to adjust your base prices per store.")

if __name__ == "__main__":
    analyze_store_performance()