"""
Electricity Cost & Battery Storage ROI Analysis Tool v2.0

Enhanced features:
- Structured file organization
- Demand and customer charges calculation
- Non-TOU profile support
- Monthly summary tables
- 10-year ROI projection
- Taiwan Dollar (TWD) currency

Directory structure:
/config/           - Configuration files (profiles.csv, calendar.csv, battery_config.json, installation_cost.json)
/input/            - User input files (consumption_hourly.csv)
/output/           - Generated reports and analysis
/output/reports/   - Text and CSV reports
/output/charts/    - Visualization charts

Installation:
pip install pandas numpy matplotlib openpyxl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def setup_directories():
    """Create organized directory structure"""
    dirs = [
        'config',
        'input',
        'output',
        'output/reports',
        'output/charts'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    return dirs

# ============================================================================
# CONFIGURATION FILES
# ============================================================================

DEFAULT_BATTERY_CONFIG = {
    'battery_capacity_kwh': 100,
    'battery_efficiency': 0.90,
    'max_charge_rate_kw': 50,
    'max_discharge_rate_kw': 50,
    'battery_cost_twd': 1500000,  # ~50k USD at 30 TWD/USD
    'battery_lifetime_years': 10,
    'annual_maintenance_pct': 0.02,
    'discount_rate': 0.05
}

DEFAULT_INSTALLATION_CONFIG = {
    'labor_cost_per_kwh_twd': 3000,
    'infrastructure_base_twd': 150000,
    'permits_and_inspection_twd': 50000,
    'contingency_pct': 0.10
}

def load_config(filename, default_config, config_dir='config'):
    """Load configuration from file or create default"""
    filepath = Path(config_dir) / filename
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        save_config(filename, default_config, config_dir)
        return default_config

def save_config(filename, config, config_dir='config'):
    """Save configuration to file"""
    filepath = Path(config_dir) / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"✓ Configuration saved to {filepath}")

def update_config(config_type='battery'):
    """Interactive configuration update"""
    if config_type == 'battery':
        config = load_config('battery_config.json', DEFAULT_BATTERY_CONFIG)
    else:
        config = load_config('installation_cost.json', DEFAULT_INSTALLATION_CONFIG)
    
    print(f"\n=== Current {config_type.title()} Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value:,.2f}" if isinstance(value, (int, float)) else f"{key}: {value}")
    
    print("\nEnter new values (press Enter to keep current value):")
    for key in config.keys():
        new_val = input(f"{key} [{config[key]}]: ").strip()
        if new_val:
            try:
                config[key] = float(new_val)
            except ValueError:
                print(f"Invalid value, keeping {config[key]}")
    
    filename = f'{config_type}_config.json'
    save_config(filename, config)
    return config

# ============================================================================
# DATA LOADING
# ============================================================================

def load_profiles(filepath='config/profiles.csv'):
    """Load electricity rate profiles (both TOU and Non-TOU)"""
    df = pd.read_csv(filepath)
    # Handle both "Non-TOU rate" and "Non-TOU" naming
    df['Type'] = df['Type'].str.replace(' rate', '', regex=False)
    return df

def load_calendar(filepath='config/calendar.csv'):
    """Load calendar with seasons and day types"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_consumption(filepath='input/consumption_hourly.csv'):
    """Load hourly consumption template"""
    df = pd.read_csv(filepath)
    # Check if first row is headers or data
    if 'Monday' not in df.columns:
        # Try reading with skiprows
        df = pd.read_csv(filepath, skiprows=1)
    return df

# ============================================================================
# RATE MATCHING (TOU)
# ============================================================================

def get_hourly_rate_tou(hour, day_type, season, profile_df):
    """Get TOU electricity rate for specific hour, day type, and season"""
    hour_frac = hour / 24.0
    
    # Filter by season (match both specific season and "All Year")
    season_mask = (profile_df['Season'] == season) | (profile_df['Season'] == 'All Year')
    
    # Filter by day type
    if day_type == 'Weekday':
        day_mask = profile_df['Day type'] == 'Weekday'
    elif day_type == 'Saturday':
        day_mask = (profile_df['Day type'] == 'Saturday') | (profile_df['Day type'] == 'Weekend & Off-Peak Day')
    else:  # Sunday & Off-Peak Day
        day_mask = (profile_df['Day type'] == 'Sunday & Off-Peak Day') | (profile_df['Day type'] == 'Weekend & Off-Peak Day')
    
    filtered = profile_df[season_mask & day_mask].copy()
    
    if filtered.empty:
        return None
    
    # Convert time to fractions
    def time_to_frac(t):
        if pd.isna(t) or t == '0:00':
            return 0.0
        if isinstance(t, (int, float)):
            return float(t)
        parts = str(t).split(':')
        return int(parts[0]) / 24.0 + int(parts[1]) / (24.0 * 60.0)
    
    filtered['start_frac'] = filtered['Start Time'].apply(time_to_frac)
    filtered['end_frac'] = filtered['End Time'].apply(time_to_frac)
    
    # Find matching time slot
    for _, row in filtered.iterrows():
        start = row['start_frac']
        end = row['end_frac']
        
        if start == 0.0 and end == 0.0:
            return row['Price']
        
        if end == 0.0 and start > 0:
            end = 1.0
        
        if start <= hour_frac < end:
            return row['Price']
    
    return None

# ============================================================================
# RATE MATCHING (Non-TOU)
# ============================================================================

def get_rate_non_tou(monthly_consumption_kwh, season, profile_df):
    """Get Non-TOU rate based on consumption tier"""
    # Filter by season
    season_mask = (profile_df['Season'] == season) | (profile_df['Season'] == 'All Year') | (profile_df['Season'].isna())
    filtered = profile_df[season_mask & (profile_df['Charge'] == 'Energy')].copy()
    
    if filtered.empty:
        return None
    
    # Find applicable tier
    for _, row in filtered.iterrows():
        # Convert to numeric, handling both time strings and numeric values
        try:
            start = float(row['Start Time']) if pd.notna(row['Start Time']) else 0
        except (ValueError, TypeError):
            start = 0
        
        try:
            end = float(row['End Time']) if pd.notna(row['End Time']) else float('inf')
        except (ValueError, TypeError):
            end = float('inf')
        
        # Check if consumption falls in this tier
        if start <= monthly_consumption_kwh < end or (end == float('inf') and monthly_consumption_kwh >= start):
            return row['Price']
    
    return None

# ============================================================================
# DEMAND & CUSTOMER CHARGES
# ============================================================================

def calculate_monthly_demand_charge(profile_df, season, peak_demand_kw):
    """Calculate monthly demand charge based on peak demand"""
    demand_charges = profile_df[
        (profile_df['Charge'] == 'Demand') &
        ((profile_df['Season'] == season) | (profile_df['Season'] == 'All Year'))
    ]
    
    total_demand_charge = 0
    for _, row in demand_charges.iterrows():
        if 'Customer charge' in str(row['Capacity category']):
            # Fixed customer charge
            total_demand_charge += row['Price']
        elif 'Contracted Demand' in str(row['Capacity category']):
            # Demand charge per kW
            total_demand_charge += peak_demand_kw * row['Price']
    
    return total_demand_charge

def calculate_customer_charge(profile_df):
    """Calculate fixed monthly customer charge"""
    customer_charges = profile_df[profile_df['Charge'] == 'Customer']
    if customer_charges.empty:
        return 0
    return customer_charges['Price'].sum()

# ============================================================================
# ANNUAL COST CALCULATION
# ============================================================================

def calculate_annual_cost(profile_contract, profile_type, profile_option, 
                         profiles_df, calendar_df, consumption_df):
    """Calculate total annual electricity cost"""
    
    # Filter profile
    profile = profiles_df[
        (profiles_df['Contract'] == profile_contract) &
        (profiles_df['Type'] == profile_type) &
        (profiles_df['Option'] == str(profile_option))
    ].copy()
    
    if profile.empty:
        raise ValueError(f"Profile not found: {profile_contract}, {profile_type}, {profile_option}")
    
    is_tou = profile_type == 'TOU'
    
    # Create 8760-hour dataframe
    results = []
    
    for _, day in calendar_df.iterrows():
        date = day['Date']
        day_type = day['DayType']
        season = day['Season']
        
        # Get consumption pattern for this day
        if day_type == 'Weekday':
            consumption = consumption_df[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean(axis=1).values
        elif day_type == 'Saturday':
            consumption = consumption_df['Saturday'].values
        else:  # Sunday & Off-Peak Day
            consumption = consumption_df['Sunday'].values
        
        # Calculate hourly costs
        for hour in range(24):
            if is_tou:
                # TOU: hourly rates
                energy_profile = profile[profile['Charge'] == 'Energy']
                rate = get_hourly_rate_tou(hour, day_type, season, energy_profile)
            else:
                # Non-TOU: will calculate monthly tier later
                rate = None
            
            kwh = consumption[hour]
            cost = kwh * rate if rate else 0
            
            results.append({
                'DateTime': datetime.combine(date, datetime.min.time()) + timedelta(hours=hour),
                'Date': date,
                'Month': date.month,
                'Hour': hour,
                'DayType': day_type,
                'Season': season,
                'Consumption_kWh': kwh,
                'Rate_per_kWh': rate,
                'Energy_Cost': cost,
                'Demand_Cost': 0,  # Will calculate monthly
                'Customer_Cost': 0  # Will calculate monthly
            })
    
    df = pd.DataFrame(results)
    
    # Add monthly demand and customer charges
    for month in range(1, 13):
        month_mask = df['Month'] == month
        month_data = df[month_mask]
        
        if month_data.empty:
            continue
        
        # Get season for this month (use first day's season)
        season = month_data.iloc[0]['Season']
        
        # Calculate peak demand (max hourly consumption)
        peak_demand_kw = month_data['Consumption_kWh'].max()
        
        # Calculate charges
        if is_tou:
            demand_charge = calculate_monthly_demand_charge(profile, season, peak_demand_kw)
            customer_charge = calculate_customer_charge(profile)
        else:
            # Non-TOU: calculate based on monthly total
            monthly_kwh = month_data['Consumption_kWh'].sum()
            energy_profile = profile[profile['Charge'] == 'Energy']
            rate = get_rate_non_tou(monthly_kwh, season, energy_profile)
            
            if rate:
                df.loc[month_mask, 'Rate_per_kWh'] = rate
                df.loc[month_mask, 'Energy_Cost'] = df.loc[month_mask, 'Consumption_kWh'] * rate
            
            demand_charge = calculate_monthly_demand_charge(profile, season, peak_demand_kw)
            customer_charge = calculate_customer_charge(profile)
        
        # Distribute monthly charges across hours (for reporting purposes)
        hours_in_month = month_mask.sum()
        df.loc[month_mask, 'Demand_Cost'] = demand_charge / hours_in_month
        df.loc[month_mask, 'Customer_Cost'] = customer_charge / hours_in_month
    
    # Total cost
    df['Total_Cost'] = df['Energy_Cost'] + df['Demand_Cost'] + df['Customer_Cost']
    
    return df

# ============================================================================
# BATTERY ARBITRAGE (Off-Peak Charging Only)
# ============================================================================

def simulate_battery_arbitrage(annual_df, config):
    """Battery arbitrage: charge during off-peak, discharge during peak"""
    df = annual_df.copy()
    
    capacity = config['battery_capacity_kwh']
    efficiency = config['battery_efficiency']
    max_charge = config['max_charge_rate_kw']
    max_discharge = config['max_discharge_rate_kw']
    
    # Add battery columns
    df['Battery_Charge_kWh'] = 0.0
    df['Battery_Discharge_kWh'] = 0.0
    df['Grid_Import_kWh'] = df['Consumption_kWh'].copy()
    df['Energy_Cost_With_Battery'] = df['Energy_Cost'].copy()
    
    # Process each day
    for date in df['Date'].unique():
        day_mask = df['Date'] == date
        day_data = df[day_mask].copy()
        
        # Skip if no rate data (e.g., Non-TOU profiles with None rates)
        if day_data['Rate_per_kWh'].isna().all() or (day_data['Rate_per_kWh'] == 0).all():
            continue
        
        # Filter out None/NaN rates
        valid_rates = day_data[day_data['Rate_per_kWh'].notna() & (day_data['Rate_per_kWh'] > 0)]
        
        if len(valid_rates) < 2:
            continue
        
        # Sort by rate to find cheapest (off-peak) and most expensive (peak) hours
        sorted_by_rate = valid_rates.sort_values('Rate_per_kWh')
        
        # Charge at cheapest off-peak hour
        cheapest_idx = sorted_by_rate.index[0]
        cheapest_rate = sorted_by_rate.iloc[0]['Rate_per_kWh']
        
        charge_amount = min(capacity, max_charge)
        df.loc[cheapest_idx, 'Battery_Charge_kWh'] = charge_amount
        df.loc[cheapest_idx, 'Grid_Import_kWh'] += charge_amount
        df.loc[cheapest_idx, 'Energy_Cost_With_Battery'] += charge_amount * cheapest_rate
        
        # Discharge at most expensive peak hour
        expensive_idx = sorted_by_rate.index[-1]
        expensive_rate = sorted_by_rate.iloc[-1]['Rate_per_kWh']
        
        discharge_amount = min(charge_amount * efficiency, max_discharge)
        df.loc[expensive_idx, 'Battery_Discharge_kWh'] = discharge_amount
        df.loc[expensive_idx, 'Grid_Import_kWh'] -= discharge_amount
        df.loc[expensive_idx, 'Energy_Cost_With_Battery'] -= discharge_amount * expensive_rate
    
    # Total cost with battery (demand and customer charges remain the same)
    df['Total_Cost_With_Battery'] = df['Energy_Cost_With_Battery'] + df['Demand_Cost'] + df['Customer_Cost']
    df['Savings'] = df['Total_Cost'] - df['Total_Cost_With_Battery']
    
    return df

# ============================================================================
# MONTHLY SUMMARY TABLE
# ============================================================================

def generate_monthly_summary(annual_df):
    """Generate monthly summary table"""
    monthly = annual_df.groupby('Month').agg({
        'Consumption_kWh': 'sum',
        'Total_Cost': 'sum',
        'Total_Cost_With_Battery': 'sum',
        'Savings': 'sum'
    }).reset_index()
    
    monthly['Savings_Pct'] = (monthly['Savings'] / monthly['Total_Cost'] * 100).round(2)
    
    # Add month names
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly['Month_Name'] = monthly['Month'].apply(lambda x: month_names[x-1])
    
    # Add total column
    total_row = pd.DataFrame([{
        'Month': 13,
        'Month_Name': 'TOTAL',
        'Consumption_kWh': monthly['Consumption_kWh'].sum(),
        'Total_Cost': monthly['Total_Cost'].sum(),
        'Total_Cost_With_Battery': monthly['Total_Cost_With_Battery'].sum(),
        'Savings': monthly['Savings'].sum(),
        'Savings_Pct': (monthly['Savings'].sum() / monthly['Total_Cost'].sum() * 100).round(2)
    }])
    
    monthly = pd.concat([monthly, total_row], ignore_index=True)
    
    return monthly

# ============================================================================
# 10-YEAR ROI PROJECTION
# ============================================================================

def calculate_10year_roi(annual_savings, battery_config, installation_config):
    """Calculate 10-year ROI projection"""
    
    # Costs
    battery_cost = battery_config['battery_cost_twd']
    capacity = battery_config['battery_capacity_kwh']
    
    labor_cost = capacity * installation_config['labor_cost_per_kwh_twd']
    infrastructure = installation_config['infrastructure_base_twd']
    permits = installation_config['permits_and_inspection_twd']
    subtotal = battery_cost + labor_cost + infrastructure + permits
    contingency = subtotal * installation_config['contingency_pct']
    total_investment = subtotal + contingency
    
    annual_maintenance = total_investment * battery_config['annual_maintenance_pct']
    discount_rate = battery_config['discount_rate']
    
    # Year-by-year projection
    projection = []
    cumulative_savings = 0
    
    for year in range(0, 11):
        if year == 0:
            # Initial investment
            projection.append({
                'Year': 0,
                'Investment': -total_investment,
                'Annual_Savings': 0,
                'Maintenance': 0,
                'Net_Cash_Flow': -total_investment,
                'Discounted_Cash_Flow': -total_investment,
                'Cumulative_Savings': -total_investment
            })
        else:
            net_benefit = annual_savings - annual_maintenance
            discounted = net_benefit / ((1 + discount_rate) ** year)
            cumulative_savings += discounted
            
            projection.append({
                'Year': year,
                'Investment': 0,
                'Annual_Savings': annual_savings,
                'Maintenance': annual_maintenance,
                'Net_Cash_Flow': net_benefit,
                'Discounted_Cash_Flow': discounted,
                'Cumulative_Savings': cumulative_savings - total_investment
            })
    
    df_projection = pd.DataFrame(projection)
    
    # Calculate metrics
    npv = df_projection['Discounted_Cash_Flow'].sum()
    simple_payback = total_investment / annual_savings if annual_savings > 0 else float('inf')
    
    roi_metrics = {
        'Total_Investment_TWD': total_investment,
        'Battery_Cost_TWD': battery_cost,
        'Installation_Cost_TWD': labor_cost + infrastructure + permits + contingency,
        'Annual_Savings_TWD': annual_savings,
        'Annual_Maintenance_TWD': annual_maintenance,
        'Simple_Payback_Years': simple_payback,
        'NPV_10Years_TWD': npv,
        'ROI_10Years_Pct': (npv / total_investment * 100) if total_investment > 0 else 0
    }
    
    return df_projection, roi_metrics

# ============================================================================
# REPORTING
# ============================================================================

def generate_reports(annual_df, monthly_summary, roi_projection, roi_metrics, profile_name):
    """Generate comprehensive reports"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f'output/reports/analysis_{timestamp}'
    
    # Monthly summary
    monthly_summary.to_csv(f'{prefix}_monthly_summary.csv', index=False, encoding='utf-8-sig')
    
    # 10-year projection
    roi_projection.to_csv(f'{prefix}_10year_projection.csv', index=False, encoding='utf-8-sig')
    
    # Full hourly data
    annual_df.to_csv(f'{prefix}_hourly_8760.csv', index=False, encoding='utf-8-sig')
    
    # ROI Report
    with open(f'{prefix}_roi_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BATTERY STORAGE ROI ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Profile: {profile_name}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=" * 80 + "\n")
        f.write("INVESTMENT BREAKDOWN (TWD)\n")
        f.write("=" * 80 + "\n")
        for key, value in roi_metrics.items():
            if 'TWD' in key or 'Investment' in key or 'Cost' in key or 'Savings' in key or 'Maintenance' in key:
                f.write(f"{key:.<60} {value:>15,.0f}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("RETURN ON INVESTMENT METRICS\n")
        f.write("=" * 80 + "\n")
        for key, value in roi_metrics.items():
            if 'Payback' in key or 'NPV' in key or 'ROI' in key:
                if 'Pct' in key:
                    f.write(f"{key:.<60} {value:>14,.2f}%\n")
                elif 'Years' in key:
                    f.write(f"{key:.<60} {value:>10,.1f} years\n")
                else:
                    f.write(f"{key:.<60} {value:>15,.0f}\n")
    
    print(f"\n✓ Reports generated in output/reports/")
    print(f"  - {prefix}_monthly_summary.csv")
    print(f"  - {prefix}_10year_projection.csv")
    print(f"  - {prefix}_roi_report.txt")
    
    # Create charts
    create_charts(monthly_summary, roi_projection, roi_metrics, timestamp)

def create_charts(monthly_df, roi_projection, roi_metrics, timestamp):
    """Create visualization charts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Monthly cost comparison
    months = monthly_df[monthly_df['Month'] != 13]['Month_Name']
    cost_baseline = monthly_df[monthly_df['Month'] != 13]['Total_Cost']
    cost_battery = monthly_df[monthly_df['Month'] != 13]['Total_Cost_With_Battery']
    
    x = np.arange(len(months))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, cost_baseline, width, label='Without Battery', alpha=0.8)
    axes[0, 0].bar(x + width/2, cost_battery, width, label='With Battery', alpha=0.8)
    axes[0, 0].set_title('Monthly Electricity Cost Comparison (TWD)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Cost (TWD)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(months, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Monthly savings percentage
    savings_pct = monthly_df[monthly_df['Month'] != 13]['Savings_Pct']
    axes[0, 1].bar(months, savings_pct, color='green', alpha=0.7)
    axes[0, 1].set_title('Monthly Energy Savings %', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Savings (%)')
    axes[0, 1].set_xticklabels(months, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. 10-year cumulative savings
    years = roi_projection['Year']
    cumulative = roi_projection['Cumulative_Savings'] / 1000  # Convert to thousands
    axes[1, 0].plot(years, cumulative, marker='o', linewidth=2, markersize=6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].fill_between(years, 0, cumulative, where=(cumulative >= 0), alpha=0.3, color='green', label='Profit')
    axes[1, 0].fill_between(years, 0, cumulative, where=(cumulative < 0), alpha=0.3, color='red', label='Loss')
    axes[1, 0].set_title('10-Year Cumulative Savings (Net Present Value)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Cumulative Savings (Thousand TWD)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ROI Summary
    axes[1, 1].axis('off')
    roi_text = f"""
    ROI SUMMARY (10 Years)
    {'='*50}
    
    Total Investment:     {roi_metrics['Total_Investment_TWD']:>15,.0f} TWD
    Annual Savings:       {roi_metrics['Annual_Savings_TWD']:>15,.0f} TWD
    
    Simple Payback:       {roi_metrics['Simple_Payback_Years']:>10,.1f} years
    Net Present Value:    {roi_metrics['NPV_10Years_TWD']:>15,.0f} TWD
    ROI (10 years):       {roi_metrics['ROI_10Years_Pct']:>14,.1f}%
    
    {'='*50}
    """
    axes[1, 1].text(0.1, 0.5, roi_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    output_path = f'output/charts/analysis_{timestamp}_charts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - {output_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("ELECTRICITY COST & BATTERY ROI ANALYSIS TOOL v2.0")
    print("=" * 80)
    
    # Setup directories
    print("\n1. Setting up directories...")
    setup_directories()
    print("   ✓ Directory structure created")
    
    # Load configurations
    print("\n2. Loading configurations...")
    battery_config = load_config('battery_config.json', DEFAULT_BATTERY_CONFIG)
    installation_config = load_config('installation_cost.json', DEFAULT_INSTALLATION_CONFIG)
    
    update = input("\nUpdate configurations? (y/n): ").strip().lower()
    if update == 'y':
        battery_config = update_config('battery')
        installation_config = update_config('installation')
    
    # Load data
    print("\n3. Loading data files...")
    try:
        profiles = load_profiles()
        calendar = load_calendar()
        consumption = load_consumption()
        print(f"   ✓ Loaded {len(profiles)} rate entries")
        print(f"   ✓ Loaded {len(calendar)} calendar days")
        print(f"   ✓ Loaded {len(consumption)} hourly consumption values")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nExpected file locations:")
        print("  - config/profiles.csv")
        print("  - config/calendar.csv")
        print("  - input/consumption_hourly.csv")
        return
    
    # Select profile
    print("\n4. Select electricity rate profile:")
    contracts = profiles['Contract'].unique()
    for i, c in enumerate(contracts, 1):
        print(f"   {i}. {c}")
    
    contract_idx = int(input("\nEnter contract number: ")) - 1
    profile_contract = contracts[contract_idx]
    
    profile_subset = profiles[profiles['Contract'] == profile_contract]
    types = profile_subset['Type'].unique()
    print(f"\nAvailable types: {', '.join(types)}")
    profile_type = input("Enter type (TOU or Non-TOU): ").strip()
    
    options = profile_subset[profile_subset['Type'] == profile_type]['Option'].unique()
    print(f"Available options: {', '.join(map(str, options))}")
    profile_option = input("Enter option: ").strip()
    
    profile_name = f"{profile_contract} - {profile_type} - Option {profile_option}"
    
    # Calculate annual cost
    print("\n5. Calculating annual electricity costs...")
    annual_df = calculate_annual_cost(
        profile_contract, profile_type, profile_option,
        profiles, calendar, consumption
    )
    print(f"   ✓ Calculated 8,760 hourly costs")
    
    # Simulate battery
    print("\n6. Simulating battery arbitrage (off-peak charging)...")
    annual_df_battery = simulate_battery_arbitrage(annual_df, battery_config)
    print(f"   ✓ Battery simulation complete")
    
    # Generate monthly summary
    print("\n7. Generating monthly summary...")
    monthly_summary = generate_monthly_summary(annual_df_battery)
    
    # Calculate 10-year ROI
    print("\n8. Calculating 10-year ROI projection...")
    annual_savings = monthly_summary[monthly_summary['Month'] == 13]['Savings'].values[0]
    roi_projection, roi_metrics = calculate_10year_roi(annual_savings, battery_config, installation_config)
    
    # Generate reports
    print("\n9. Generating reports and charts...")
    generate_reports(annual_df_battery, monthly_summary, roi_projection, roi_metrics, profile_name)
    
    # Display summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nProfile: {profile_name}")
    print(f"\nANNUAL COSTS (TWD):")
    print(f"  Baseline (without battery):  {monthly_summary[monthly_summary['Month']==13]['Total_Cost'].values[0]:>15,.0f}")
    print(f"  With battery:                {monthly_summary[monthly_summary['Month']==13]['Total_Cost_With_Battery'].values[0]:>15,.0f}")
    print(f"  Annual savings:              {annual_savings:>15,.0f}")
    print(f"  Savings percentage:          {monthly_summary[monthly_summary['Month']==13]['Savings_Pct'].values[0]:>14,.1f}%")
    
    print(f"\nINVESTMENT (TWD):")
    print(f"  Total investment:            {roi_metrics['Total_Investment_TWD']:>15,.0f}")
    print(f"  - Battery cost:              {roi_metrics['Battery_Cost_TWD']:>15,.0f}")
    print(f"  - Installation cost:         {roi_metrics['Installation_Cost_TWD']:>15,.0f}")
    
    print(f"\nROI METRICS:")
    print(f"  Simple payback period:       {roi_metrics['Simple_Payback_Years']:>10,.1f} years")
    print(f"  NPV (10 years):              {roi_metrics['NPV_10Years_TWD']:>15,.0f} TWD")
    print(f"  ROI (10 years):              {roi_metrics['ROI_10Years_Pct']:>14,.1f}%")
    
    print("\n" + "=" * 80)
    print("\n✓ All reports saved in output/ directory")
    print("  - Monthly summary: output/reports/")
    print("  - 10-year projection: output/reports/")
    print("  - Charts: output/charts/")
    
    # Display monthly table
    print("\n" + "=" * 80)
    print("MONTHLY SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Month':<12} {'Consumption':<15} {'Cost (No Bat)':<18} {'Cost (With Bat)':<18} {'Savings %':<12}")
    print(f"{'':12} {'(kWh)':<15} {'(TWD)':<18} {'(TWD)':<18} {'(%)':<12}")
    print("-" * 80)
    
    for _, row in monthly_summary.iterrows():
        month_name = row['Month_Name']
        consumption = row['Consumption_kWh']
        cost_no_bat = row['Total_Cost']
        cost_with_bat = row['Total_Cost_With_Battery']
        savings_pct = row['Savings_Pct']
        
        print(f"{month_name:<12} {consumption:>14,.0f} {cost_no_bat:>17,.0f} {cost_with_bat:>17,.0f} {savings_pct:>11,.1f}")
    
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")