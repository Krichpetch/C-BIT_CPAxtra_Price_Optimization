import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def simulate_real_financials(subm_df, raw_dir='Raws'):
    """
    Simulates actual real-world grocery financials.
    Calculates Realized Demand (capped by inventory), Total Revenue, and Real Profit.
    Ignores Kaggle stockout penalties, as they are not real cash losses.
    """
    try:
        # Load necessary raw data
        sales = pd.read_csv(os.path.join(raw_dir, 'sales_history.csv'))
        pc = pd.read_csv(os.path.join(raw_dir, 'price_cost.csv'))
        inv = pd.read_csv(os.path.join(raw_dir, 'inventory.csv'))
        
        # Calculate base daily demand from history
        base_demand = sales.groupby(['store_id', 'sku_id'])['qty'].mean().reset_index()
        base_demand.rename(columns={'qty': 'base_daily_demand'}, inplace=True)
        
        # Merge auxiliary data into the submission dataframe
        df = subm_df.copy()
        df = df.merge(pc[['sku_id', 'regular_price', 'unit_cost']], on='sku_id', how='left')
        df = df.merge(base_demand, on=['store_id', 'sku_id'], how='left')
        
        # Fill missing base demands with a generic 5.0
        df['base_daily_demand'] = df['base_daily_demand'].fillna(5.0)
        
        # 1. Calculate Expected Daily Demand using Elasticity (-1.3)
        elasticity = -1.3
        df['expected_daily_demand'] = df['base_daily_demand'] * ((df['proposed_price'] / df['regular_price']) ** elasticity)
        
        # 2. Aggregate to 14-Day Totals per Store-SKU
        agg_df = df.groupby(['store_id', 'sku_id']).agg(
            total_14d_demand=('expected_daily_demand', 'sum'),
            avg_proposed_price=('proposed_price', 'mean'),
            unit_cost=('unit_cost', 'first')
        ).reset_index()
        
        # 3. Bring in Inventory Caps (You cannot sell what you do not have)
        agg_df = agg_df.merge(inv[['store_id', 'sku_id', 'on_hand']], on=['store_id', 'sku_id'], how='left')
        agg_df['on_hand'] = agg_df['on_hand'].fillna(0)
        
        # 4. Calculate Real Business Financials
        agg_df['actual_units_sold'] = np.minimum(agg_df['total_14d_demand'], agg_df['on_hand'])
        
        total_revenue = (agg_df['actual_units_sold'] * agg_df['avg_proposed_price']).sum()
        total_cogs = (agg_df['actual_units_sold'] * agg_df['unit_cost']).sum()
        real_gross_profit = total_revenue - total_cogs
        
        return total_revenue, total_cogs, real_gross_profit
        
    except Exception as e:
        return None, None, f"Error running simulation: {e}"


def compare_csv_files(file1_path, file2_path, key_columns=None, output_file='comparison_report.txt', run_financials=True):
    """
    Compare two CSV files and generate a detailed comparison report, including real profit simulation.
    """
    print("="*80)
    print("CSV COMPARISON & REAL FINANCIAL SIMULATION")
    print("="*80)
    print(f"\nFile 1 (Baseline): {file1_path}")
    print(f"File 2 (New Model): {file2_path}")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load the CSV files
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        print(f"\n✓ Successfully loaded both files")
        print(f"  File 1: {len(df1):,} rows × {len(df1.columns)} columns")
        print(f"  File 2: {len(df2):,} rows × {len(df2.columns)} columns")
    except Exception as e:
        print(f"\n✗ Error loading files: {e}")
        return
    
    # Clean column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Start building the report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("CSV COMPARISON REPORT")
    report_lines.append("="*80)
    report_lines.append(f"File 1 (Baseline): {file1_path}")
    report_lines.append(f"File 2 (New):      {file2_path}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    
    # 1. STRUCTURE COMPARISON
    report_lines.append("\n" + "="*80)
    report_lines.append("1. STRUCTURE COMPARISON")
    report_lines.append("="*80)
    
    report_lines.append(f"\nRow Count:")
    report_lines.append(f"  File 1: {len(df1):,} rows")
    report_lines.append(f"  File 2: {len(df2):,} rows")
    row_diff = len(df2) - len(df1)
    report_lines.append(f"  Difference: {row_diff:+,} rows ({(row_diff/len(df1)*100):+.2f}%)")
    
    # Column differences
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    common_cols = cols1.intersection(cols2)
    
    # Auto-detect key columns if not provided
    if key_columns is None:
        potential_keys = [col for col in common_cols if any(k in col.lower() for k in ['id', 'store', 'sku', 'date'])]
        if potential_keys:
            key_columns = potential_keys
    
    # 2. DATA COMPARISON
    report_lines.append("\n" + "="*80)
    report_lines.append("2. PRICING DATA COMPARISON")
    report_lines.append("="*80)
    
    if key_columns and all(col in common_cols for col in key_columns):
        merged = df1.merge(df2, on=key_columns, how='inner', suffixes=('_old', '_new'))
        
        if 'proposed_price_old' in merged.columns and 'proposed_price_new' in merged.columns:
            diff_mask = merged['proposed_price_old'] != merged['proposed_price_new']
            changed_rows = diff_mask.sum()
            
            report_lines.append(f"\nPrice Changes Overview:")
            report_lines.append(f"  Prices Changed: {changed_rows:,} rows out of {len(merged):,} ({(changed_rows/len(merged))*100:.1f}%)")
            
            if changed_rows > 0:
                price_diffs = merged.loc[diff_mask, 'proposed_price_new'] - merged.loc[diff_mask, 'proposed_price_old']
                report_lines.append(f"  Price Increased: {(price_diffs > 0).sum():,} rows")
                report_lines.append(f"  Price Decreased: {(price_diffs < 0).sum():,} rows")
                report_lines.append(f"  Avg Change Amount: {price_diffs.mean():+.2f} THB")
    
    # 3. REAL FINANCIAL SIMULATION (The Money Section)
    if run_financials and 'proposed_price' in df1.columns and 'proposed_price' in df2.columns:
        report_lines.append("\n" + "="*80)
        report_lines.append("3. REAL BUSINESS FINANCIAL IMPACT (14-DAY SIMULATION)")
        report_lines.append("="*80)
        
        print("\nRunning financial simulation (this takes a few seconds)...")
        rev1, cogs1, profit1 = simulate_real_financials(df1)
        rev2, cogs2, profit2 = simulate_real_financials(df2)
        
        if isinstance(profit1, str) or isinstance(profit2, str):
            report_lines.append(f"\n⚠ Could not run simulation. Error: {profit1 or profit2}")
            report_lines.append("Make sure the 'Raws/' folder exists with sales_history, price_cost, and inventory CSVs.")
        else:
            profit_diff = profit2 - profit1
            profit_pct = (profit_diff / abs(profit1)) * 100 if profit1 != 0 else 0
            
            report_lines.append("\nFile 1 (Baseline) Real Financials:")
            report_lines.append(f"  Total Revenue:       {rev1:,.0f} THB")
            report_lines.append(f"  Total COGS:         -{cogs1:,.0f} THB")
            report_lines.append(f"  Real Gross Profit:   {profit1:,.0f} THB")
            
            report_lines.append("\nFile 2 (New Model) Real Financials:")
            report_lines.append(f"  Total Revenue:       {rev2:,.0f} THB")
            report_lines.append(f"  Total COGS:         -{cogs2:,.0f} THB")
            report_lines.append(f"  Real Gross Profit:   {profit2:,.0f} THB")
            
            report_lines.append("\nFINANCIAL IMPACT (New vs Baseline):")
            report_lines.append(f"  Actual Profit Gained: {profit_diff:+,.0f} THB")
            report_lines.append(f"  ROI / Improvement:    {profit_pct:+.1f} %")
            
            if profit_diff > 0:
                report_lines.append("\n✅ The New Model generates more actual cash profit than the baseline!")
            else:
                report_lines.append("\n❌ The New Model generates less cash profit than the baseline.")

    report_lines.append(f"\n" + "="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Print to console
    print('\n'.join(report_lines))
    print(f"\n✓ Comparison report saved to: {output_file}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
if __name__ == "__main__":
    
    # REPLACE THESE PATHS with your actual submission files!
    file_baseline = 'Submissions/submission_baseline_last_price.csv'   
    file_new_model = 'Submissions/submission_chunking_v1_20260222_204423.csv' 
    
    if os.path.exists(file_baseline) and os.path.exists(file_new_model):
        compare_csv_files(
            file1_path=file_baseline,
            file2_path=file_new_model,
            output_file='real_profit_comparison_report.txt',
            run_financials=True
        )
    else:
        print("Please point the script to two valid CSV submission files.")