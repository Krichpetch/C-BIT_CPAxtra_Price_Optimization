import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. LOAD DATA & SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Fallback loader
def get_file(filename):
    if os.path.exists(filename): return filename
    elif os.path.exists(f'Raws/{filename}'): return f'Raws/{filename}'
    raise FileNotFoundError(f"Cannot find {filename}")

print("Loading Cross-Elasticity data...")
df = pd.read_csv(get_file('cross_elasticity_report.csv'))

# ==========================================
# 2. FIND A STRONG TEST CASE
# ==========================================
# We want to find a target item that has BOTH a strong Substitute and a strong Complement
# Substitutes have XEL > 0 (Positive)
# Complements have XEL < 0 (Negative)

substitutes = df[df['relationship_type'].str.contains('Substitute')]
complements = df[df['relationship_type'].str.contains('Complement')]

# Find the strongest substitute pair
top_sub = substitutes.loc[substitutes['xel_ij'].idxmax()]
# Find a strong complement for that SAME target item (if possible)
target_sku = top_sub['sku_i']
target_name = top_sub['target_name']

comps_for_target = complements[complements['sku_i'] == target_sku]
if len(comps_for_target) > 0:
    top_comp = comps_for_target.loc[comps_for_target['xel_ij'].idxmin()]
else:
    # If no exact match, just pick the strongest complement overall for demonstration
    top_comp = complements.loc[complements['xel_ij'].idxmin()]
    target_name = "Blended Basket Example"

# ==========================================
# 3. RUN THE DEMAND SHAPING SIMULATION
# ==========================================
# Baseline Scenario
base_daily_demand = 100  # Assume we sell 100 units a day normally
days_remaining = 14
base_total_demand = base_daily_demand * days_remaining

# Define our tactical price changes on the DRIVER items
sub_discount_pct = -0.20  # 20% Discount on the Substitute
comp_markup_pct = 0.20    # 20% Markup on the Complement

# Calculate the Ripple Effects (Formula: % Change in Demand = XEL * % Change in Price)
sub_effect_pct = top_sub['xel_ij'] * sub_discount_pct  # Positive XEL * Negative Price = Negative Demand
comp_effect_pct = top_comp['xel_ij'] * comp_markup_pct # Negative XEL * Positive Price = Negative Demand

# Apply the effects to our volume
sub_volume_saved = base_total_demand * abs(sub_effect_pct)
comp_volume_saved = base_total_demand * abs(comp_effect_pct)
final_demand = base_total_demand - sub_volume_saved - comp_volume_saved

print(f"\n--- XEL DEMAND SHAPING SIMULATION ---")
print(f"Target Item at Risk: {target_name} (Base 14-Day Demand: {base_total_demand} units)")
print(f"\nTACTIC 1: Discount Substitute -> {top_sub['driver_name']} by 20%")
print(f"  -> XEL Strength: {top_sub['xel_ij']:.3f}")
print(f"  -> Effect: Shifts demand away, saving {sub_volume_saved:.1f} units of Target stock.")

print(f"\nTACTIC 2: Markup Complement -> {top_comp['driver_name']} by +20%")
print(f"  -> XEL Strength: {top_comp['xel_ij']:.3f}")
print(f"  -> Effect: Chokes basket purchases, saving {comp_volume_saved:.1f} units of Target stock.")

print(f"\nFINAL RESULT: Projected demand dropped from {base_total_demand} to {final_demand:.1f} units without ever touching the Target's price!")

# ==========================================
# 4. VISUALIZE THE "ECOSYSTEM DEFENSE" (Waterfall Chart)
# ==========================================
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 6))

# Data for waterfall
categories = ['Original Demand\n(Stockout Risk)', 
              f'Shift to Substitute\n({top_sub["driver_name"]})', 
              f'Basket Choke via Complement\n({top_comp["driver_name"]})', 
              'Final Rescued\nDemand']
values = [base_total_demand, -sub_volume_saved, -comp_volume_saved, final_demand]

# Calculate running totals for waterfall bottoms
bottoms = [0, base_total_demand - sub_volume_saved, final_demand, 0]
colors = ['#4575b4', '#31a354', '#31a354', '#d73027']

# Plot bars
bars = ax.bar(categories, values, bottom=bottoms, color=colors, edgecolor='black', width=0.6)

# Formatting
ax.set_title(f'The Ecosystem Defense: Saving "{target_name}" via XEL', fontsize=16, fontweight='bold')
ax.set_ylabel('14-Day Projected Demand (Units)', fontsize=12, fontweight='bold')

# Add connection lines between bars
for i in range(len(values)-1):
    if i == 0:
        ax.plot([i, i+1], [base_total_demand, base_total_demand], color='grey', linestyle='--')
    else:
        ax.plot([i, i+1], [bottoms[i], bottoms[i]], color='grey', linestyle='--')

# Add data labels
for i, bar in enumerate(bars):
    height = values[i]
    y_pos = bottoms[i] + (height / 2) if i > 0 and i < 3 else bottoms[i] + height + 20
    text = f"{height:+.1f}" if i in [1, 2] else f"{height:.0f} units"
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, text, 
            ha='center', va='center', fontweight='bold', color='white' if i in [1,2] else 'black', fontsize=11)

plt.tight_layout()
output_file = 'xel_ecosystem_defense.png'
plt.savefig(output_file, dpi=150)
print(f"\n✅ Visual saved as: {output_file}")