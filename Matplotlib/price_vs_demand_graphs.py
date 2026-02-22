import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_predictive_elasticity_graph():
    # 1. Define the Price Strategy Points
    price_adjustments = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30])
    labels = [
        '-30% Discount', '-20% Discount', '-10% Discount', 
        'Regular Price', 
        '+10% Premium', '+20% Premium', '+30% Premium'
    ]
    
    # 2. The Core Math of your Algorithm
    base_demand = 10.0 # Example baseline: 10 units sold per day
    elasticity = -1.3  # The constant used in our optimizer
    
    # Formula: Q_new = Q_base * (P_new / P_base) ^ elasticity
    price_ratios = 1 + price_adjustments
    predicted_demand = base_demand * (price_ratios ** elasticity)

    # Calculate percentage change in volume
    volume_change = ((predicted_demand - base_demand) / base_demand) * 100

    # 3. Plotting Setup
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(13, 7))

    # Clean color mapping: Blue for Danger (Discounts), Grey for Base, Red for Protection (Premiums)
    colors = ['#313695', '#4575b4', '#74add1', '#999999', '#f46d43', '#d73027', '#a50026']

    # Draw Bars
    bars = ax.bar(labels, predicted_demand, color=colors, edgecolor='black', alpha=0.85)
    
    # Draw the smooth Predictive Trendline
    ax.plot(labels, predicted_demand, color='black', marker='o', linestyle='-', linewidth=3, markersize=8)

    # Formatting
    ax.set_title('🧠 ML Engine Prediction: How Price Hikes Stretch Inventory', fontsize=18, fontweight='bold')
    ax.set_xlabel('Pricing Decision vs. Regular Price', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Daily Demand (Units)', fontsize=12, fontweight='bold')

    # Add Text Annotations (Show Units AND % Volume Drop)
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        v_change = volume_change[i]
        
        if v_change == 0:
            text = f"{yval:.1f} units\n(Baseline)"
        elif v_change > 0:
            text = f"{yval:.1f} units\n(+{v_change:.1f}% Vol)"
        else:
            text = f"{yval:.1f} units\n({v_change:.1f}% Vol)"
            
        # Bold text directly above the bars
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.3, text, ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Highlight Zones
    ax.axvspan(3.5, 6.5, color='red', alpha=0.1, label='Demand Suppression (Algorithm stretches inventory)')
    ax.axvspan(-0.5, 2.5, color='blue', alpha=0.1, label='Stockout Danger Zone (Algorithm predicts instant drain)')

    ax.legend(loc='upper right', fontsize=12, facecolor='white', framealpha=0.9)
    ax.set_ylim(0, max(predicted_demand) + 3)

    # Save Output
    plt.tight_layout()
    output_file = 'predictive_elasticity_curve.png'
    plt.savefig(output_file, dpi=150)
    print(f"✅ Predictive Graph saved as: {output_file}")

if __name__ == "__main__":
    generate_predictive_elasticity_graph()