import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 1️⃣ Path Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'Raws')

def load_data(filename):
    raw_path = os.path.join(RAW_DIR, filename)
    local_path = os.path.join(BASE_DIR, filename)
    
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    elif os.path.exists(local_path):
        return pd.read_csv(local_path)
    else:
        raise FileNotFoundError(f"❌ File not found: {filename}")

# =========================
# 2️⃣ Data Generation/Plotting
# =========================
def generate_inventory_dashboard():
    print("Loading risk report data...")
    
    try:
        df = load_data('stockout_risk_report_ultimate.csv')
    except FileNotFoundError:
        print("Kaggle risk report not found. Please run the stockout report generation script first.")
        return
        
    # --- Map and Calculate Columns ---
    # Map the columns from the ultimate report to what the dashboard expects
    df['days_stock_will_last'] = df['days_of_cover']
    df['14_day_expected_demand'] = df['projected_avg_daily_sales'] * 14
    df['deficit'] = df['14_day_expected_demand'] - df['on_hand']

    # --- Plotting Setup ---
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📦 Inventory Health & Stockout Risk Dashboard', fontsize=22, fontweight='bold')

    # PLOT 1: Days of Stock Distribution (Histogram)
    ax1 = axes[0, 0]
    data_to_plot = df[df['days_stock_will_last'] < 14]['days_stock_will_last']
    ax1.hist(data_to_plot, bins=14, color='coral', edgecolor='black', alpha=0.8)
    ax1.set_title('Distribution of Remaining Inventory (Days of Stock)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Days of Stock Remaining')
    ax1.set_ylabel('Number of Store-SKU Combinations')
    
    out_of_bounds = len(df[df['days_stock_will_last'] >= 14])
    total_items = len(df)
    ax1.text(0.5, 0.9, f"{total_items - out_of_bounds:,} out of {total_items:,} items\nwill stock out in < 14 Days", 
             transform=ax1.transAxes, ha='center', fontsize=12, fontweight='bold', color='darkred',
             bbox=dict(facecolor='white', alpha=0.8))

    # PLOT 2: Demand vs Supply Scatter
    ax2 = axes[0, 1]
    ax2.scatter(df['on_hand'], df['14_day_expected_demand'], alpha=0.6, color='royalblue', s=30)
    max_val = max(df['on_hand'].max(), df['14_day_expected_demand'].max())
    
    ax2.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=2, label='Demand = Supply')
    ax2.fill_between([0, max_val], [0, max_val], max_val, color='red', alpha=0.1, label='Stockout Danger Zone (Demand > Stock)')
    ax2.fill_between([0, max_val], 0, [0, max_val], color='green', alpha=0.1, label='Safe Zone (Overstock)')
    
    ax2.set_title('14-Day Expected Demand vs. Current On-Hand Stock', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Current On-Hand Inventory (Units)')
    ax2.set_ylabel('14-Day Expected Demand (Units)')
    ax2.legend(loc='upper left')

    # PLOT 3: Risk Level Distribution (Horizontal Bar)
    ax3 = axes[1, 0]
    risk_counts = df['risk_level'].value_counts()
    
    # Slice the colors array in case there are fewer than 4 risk levels identified
    colors = ['#b30000', '#e34a33', '#fc8d59', '#fef0d9']
    ax3.barh(risk_counts.index, risk_counts.values, color=colors[:len(risk_counts)], edgecolor='black')
    ax3.invert_yaxis() 
    
    ax3.set_title('Count of Items by Risk Level', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Number of Store-SKU Combinations')
    ax3.set_ylabel('')
    
    for index, value in enumerate(risk_counts.values):
        if value > max(risk_counts.values) * 0.1:
            ax3.text(value - (max(risk_counts.values) * 0.05), index, f"{value:,}", va='center', ha='right', color='white', fontweight='bold', fontsize=12)
        else:
            ax3.text(value + (max(risk_counts.values) * 0.02), index, f"{value:,}", va='center', ha='left', color='black', fontweight='bold', fontsize=12)

    # PLOT 4: Top 10 Worst Stockout Deficits (Horizontal Bar)
    ax4 = axes[1, 1]
    
    top_deficits = df.sort_values('deficit', ascending=False).head(10).copy()
    top_deficits['label'] = 'Store ' + top_deficits['store_id'].astype(str) + ' | SKU ' + top_deficits['sku_id'].astype(str)
    
    ax4.barh(top_deficits['label'], top_deficits['deficit'], color='#cb181d', edgecolor='black')
    ax4.invert_yaxis()
    
    ax4.set_title('Top 10 Worst Inventory Deficits', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Unit Deficit (Expected Demand - Current On-Hand)')
    ax4.set_ylabel('')
    
    for index, row in enumerate(top_deficits.itertuples()):
        ax4.text(row.deficit - (top_deficits['deficit'].max() * 0.05), index, 
                 f"-{row.deficit:.1f} units", va='center', ha='right', color='white', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = 'inventory_dashboard.png'
    plt.savefig(output_file, dpi=100)
    print(f"\n✅ Dashboard saved as: {output_file}")

if __name__ == "__main__":
    generate_inventory_dashboard()