import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
# 2️⃣ Data Processing
# =========================

def generate_graphs():
    print("Loading data...")
    try:
        sales = load_data('sales_history.csv')
        # Load the calendar file which contains the exact 'is_payday' flags
        cal = load_data('calendar_weather.csv')
    except Exception as e:
        print(e)
        return

    # --- Feature Engineering ---
    sales['date'] = pd.to_datetime(sales['date'])
    cal['date'] = pd.to_datetime(cal['date'])
    
    sales['day_of_week'] = sales['date'].dt.day_name()
    sales['day'] = sales['date'].dt.day
    
    # Merge exact payday flags from the calendar
    sales = sales.merge(cal[['date', 'is_payday']], on='date', how='left')
    
    # Convert binary flags to readable labels
    sales['payday_label'] = sales['is_payday'].apply(lambda x: "Payday Period" if x == 1 else "Normal Day")
    
    # --- Aggregations ---
    
    # 1. Day of Week Profile
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_avg = sales.groupby('day_of_week')['qty'].mean().reindex(dow_order)
    
    # 2. Payday Impact (Using the exact calendar flags)
    payday_avg = sales.groupby('payday_label')['qty'].mean()
    
    # 3. Forecast Generation (Aug 14 - Aug 27)
    future_dates = pd.date_range(start='2025-08-14', end='2025-08-27')
    forecast = pd.DataFrame({'date': future_dates})
    forecast['day_of_week'] = forecast['date'].dt.day_name()
    
    # Merge future payday flags from calendar into the forecast
    forecast = forecast.merge(cal[['date', 'is_payday']], on='date', how='left')
    
    # Calculate simple multipliers based on history
    global_avg = sales['qty'].mean()
    dow_mult = (dow_avg / global_avg).to_dict()
    payday_mult = (payday_avg['Payday Period'] / payday_avg['Normal Day'])
    
    def predict_score(row):
        score = dow_mult.get(row['day_of_week'], 1.0)
        # Apply multiplier strictly if the calendar marked it as a payday (1)
        if row['is_payday'] == 1:
            score *= payday_mult
        return score

    forecast['demand_score'] = forecast.apply(predict_score, axis=1)
    
    # NEW: Calculate the Actual Estimated Qty for the future days
    forecast['estimated_qty'] = forecast['demand_score'] * global_avg

    # =========================
    # 3️⃣ Plotting
    # =========================
    
    print("Generating Dashboard...")
    plt.style.use('ggplot') 
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📅 Seasonality & Sales Opportunity Dashboard', fontsize=18, fontweight='bold')

    # --- PLOT 1: Weekly Rhythm (Bar Chart) ---
    ax1 = axes[0, 0]
    dow_avg.plot(kind='bar', color='skyblue', ax=ax1, edgecolor='black')
    ax1.set_title('Average Sales Volume by Day of Week')
    ax1.set_ylabel('Avg Qty Sold')
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=45)
    # Add labels on bars
    for i, v in enumerate(dow_avg):
        ax1.text(i, v + 0.1, f"{v:.1f}", ha='center', fontsize=10, fontweight='bold')

    # --- PLOT 2: Payday Effect (Bar Chart) ---
    ax2 = axes[0, 1]
    colors = ['lightblue', 'red']
    payday_avg.plot(kind='bar', color=colors, ax=ax2, edgecolor='black', alpha=0.8)
    ax2.set_title('The "Payday Spike" Effect')
    ax2.set_ylabel('Avg Qty Sold')
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=0)
    
    # NEW: Add sales numbers above Payday bars
    for i, v in enumerate(payday_avg):
        ax2.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=12, fontweight='bold')
    
    # Calculate Lift %
    lift = (payday_avg['Payday Period'] - payday_avg['Normal Day']) / payday_avg['Normal Day'] * 100
    ax2.text(0.5, payday_avg.max()*0.5, f"+{lift:.1f}% LIFT", 
             ha='center', fontsize=14, fontweight='bold', color='darkred', 
             bbox=dict(facecolor='white', alpha=0.8))

    # --- PLOT 3: Forecast Timeline (Line Chart) ---
    ax3 = axes[1, 0]
    ax3.plot(forecast['date'], forecast['demand_score'], marker='o', linestyle='-', linewidth=2, color='darkgreen')
    ax3.set_title('Predicted Demand Strength (Aug 14 - 27)')
    ax3.set_ylabel('Demand Index (1.0 = Average)')
    ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Average Day')
    
    # NEW: Add Estimated Quantity numbers above the line points
    for x, y, est_qty in zip(forecast['date'], forecast['demand_score'], forecast['estimated_qty']):
        ax3.text(x, y + 0.02, f"{est_qty:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
    
    # Highlight Payday start
    first_future_payday = forecast[forecast['is_payday'] == 1]['date'].min()
    if pd.notna(first_future_payday):
        ax3.axvline(first_future_payday, color='gold', linestyle='--', linewidth=3, label='Payday Starts')
    
    ax3.legend()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax3.tick_params(axis='x', rotation=45)

    # --- PLOT 4: Strategy Heatmap (Scatter) ---
    ax4 = axes[1, 1]
    
    # Define colors based on score
    conditions = [
        (forecast['demand_score'] > 1.1),  # High Traffic
        (forecast['demand_score'] < 0.95)  # Low Traffic
    ]
    choices = ['red', 'blue'] # Red = Hot/Promo, Blue = Cold/Clearance
    forecast['color'] = np.select(conditions, choices, default='gray')
    
    ax4.scatter(forecast['date'], [1]*len(forecast), s=forecast['demand_score']*500, c=forecast['color'], alpha=0.6)
    ax4.set_title('Action Plan: Traffic Intensity')
    ax4.set_yticks([]) # Hide y-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax4.tick_params(axis='x', rotation=45)
    
    # NEW: Add Estimated Quantity numbers directly above the Heatmap scatter circles
    for x, y, est_qty in zip(forecast['date'], [1]*len(forecast), forecast['estimated_qty']):
        ax4.text(x, y + 0.05, f"{est_qty:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='High Traffic (Maximize Availability)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Normal Traffic'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Low Traffic (Deep Discounts)')
    ]
    ax4.legend(handles=legend_elements, loc='upper left')

    # --- Save ---
    plt.tight_layout()
    output_file = 'seasonality_dashboard.png'
    plt.savefig(output_file, dpi=100)
    print(f"\n✅ Dashboard saved as: {output_file}")

if __name__ == "__main__":
    generate_graphs()