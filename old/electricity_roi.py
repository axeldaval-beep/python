"""
Electricity Cost & Battery Storage ROI Analysis Tool

This tool calculates annual electricity costs based on TOU rates and consumption patterns,
simulates battery arbitrage, and provides ROI analysis.

Required CSV files:
1. profiles.csv - Rate profiles with TOU pricing
2. calendar.csv - 365-day calendar with seasons and day types
3. consumption_hourly.csv - 24-hour consumption template for each day of week

Installation:
pip install pandas numpy matplotlib openpyxl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_FILE = 'battery_config.json'

DEFAULT_CONFIG = {
    'battery_capacity_kwh': 100,
    'battery_efficiency': 0.90,
    'max_charge_rate_kw': 50,
    'max_discharge_rate_kw': 50,
    'battery_cost_usd': 50000,
    'battery_lifetime_years': 10,
    'annual_maintenance_pct': 0.02,
    'discount_rate': 0.05
}

def load_config():
    """Load battery configuration from file or create default"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

def save_config(config):
    """Save battery configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {CONFIG_FILE}")

def update_config():
    """Interactive configuration update"""
    config = load_config()
    print("\n=== Current Battery Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    print("\nEnter new values (press Enter to keep current value):")
    for key in config.keys():
        new_val = input(f"{key} [{config[key]}]: ").strip()
        if new_val:
            try:
                config[key] = float(new_val)
            except ValueError:
                print(f"Invalid value, keeping {config[key]}")
    
    save_config(config)
    return config

# ============================================================================
# DATA LOADING
# ============================================================================

def load_profiles(filepath='profiles.csv'):
    """Load electricity rate profiles"""
    df = pd.read_csv(filepath)
    # Filter only Energy charges for TOU profiles
    df = df[(df['Charge'] == 'Energy') & (df['Type'] == 'TOU')].copy()
    return df

def load_calendar(filepath='calendar.csv'):
    """Load calendar with seasons and day types"""
    df = pd.read_csv(filepath)
    print(f"   Calendar columns: {list(df.columns)}")  # DEBUG LINE
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_consumption(filepath='consumption_hourly.csv'):
    """Load hourly consumption template"""
    df = pd.read_csv(filepath)
    return df

# ============================================================================
# RATE MATCHING
# ============================================================================

def get_hourly_rate(hour, day_type, season, profile_df):
    """
    Get electricity rate for a specific hour, day type, and season
    
    Args:
        hour: 0-23
        day_type: 'Weekday', 'Saturday', 'Sunday & Off-Peak Day', 'Weekend & Off-Peak Day'
        season: 'Summer' or 'Non-Summer'
        profile_df: Filtered profile dataframe for specific contract/type/option
    
    Returns:
        rate in currency per kWh
    """
    # Convert hour to time format for matching
    hour_frac = hour / 24.0
    
    # Filter by season (match both specific season and "All Year")
    season_mask = (profile_df['Season'] == season) | (profile_df['Season'] == 'All Year')
    
    # Filter by day type
    # Handle both specific day types and combined weekend types
    if day_type == 'Weekday':
        day_mask = profile_df['Day type'] == 'Weekday'
    elif day_type == 'Saturday':
        day_mask = (profile_df['Day type'] == 'Saturday') | (profile_df['Day type'] == 'Weekend & Off-Peak Day')
    else:  # Sunday & Off-Peak Day
        day_mask = (profile_df['Day type'] == 'Sunday & Off-Peak Day') | (profile_df['Day type'] == 'Weekend & Off-Peak Day')
    
    # Combine filters
    filtered = profile_df[season_mask & day_mask].copy()
    
    if filtered.empty:
        return None
    
    # Convert time strings to fractions if needed
    def time_to_frac(t):
        if pd.isna(t) or t == '0:00':
            return 0.0
        if isinstance(t, (int, float)):
            return float(t)
        # Parse time string "HH:MM"
        parts = str(t).split(':')
        return int(parts[0]) / 24.0 + int(parts[1]) / (24.0 * 60.0)
    
    filtered['start_frac'] = filtered['Start Time'].apply(time_to_frac)
    filtered['end_frac'] = filtered['End Time'].apply(time_to_frac)
    
    # Find matching time slot
    for _, row in filtered.iterrows():
        start = row['start_frac']
        end = row['end_frac']
        
        # Handle all-day rate (0:00 to 0:00)
        if start == 0.0 and end == 0.0:
            return row['Price']
        
        # Handle midnight wraparound (e.g., 22:00 to 0:00)
        if end == 0.0 and start > 0:
            end = 1.0  # Treat as end of day
        
        # Check if hour falls in this time slot
        if start <= hour_frac < end:
            return row['Price']
    
    return None

# ============================================================================
# ANNUAL COST CALCULATION
# ============================================================================

def calculate_annual_cost(profile_contract, profile_type, profile_option, 
                         profiles_df, calendar_df, consumption_df):
    """
    Calculate total annual electricity cost without battery
    
    Returns:
        DataFrame with 8760 rows (365 days × 24 hours)
    """
    # Filter profile
    profile = profiles_df[
        (profiles_df['Contract'] == profile_contract) &
        (profiles_df['Type'] == profile_type) &
        (profiles_df['Option'] == profile_option)
    ].copy()
    
    if profile.empty:
        raise ValueError(f"Profile not found: {profile_contract}, {profile_type}, {profile_option}")
    
    # Create 8760-hour dataframe
    results = []
    
    for _, day in calendar_df.iterrows():
        date = day['Date']
        day_type = day['DayType']
        season = day['Season']
        day_name = day['DayName']
        
        # Get consumption pattern for this day
        if day_type == 'Weekday':
            # Use average of Mon-Fri
            consumption = consumption_df[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean(axis=1).values
        elif day_type == 'Saturday':
            consumption = consumption_df['Saturday'].values
        else:  # Sunday & Off-Peak Day
            consumption = consumption_df['Sunday'].values
        
        # Calculate cost for each hour of this day
        for hour in range(24):
            rate = get_hourly_rate(hour, day_type, season, profile)
            
            if rate is None:
                print(f"Warning: No rate found for {date} hour {hour}, {day_type}, {season}")
                rate = 0
            
            kwh = consumption[hour]
            cost = kwh * rate
            
            results.append({
                'DateTime': datetime.combine(date, datetime.min.time()) + timedelta(hours=hour),
                'Date': date,
                'Hour': hour,
                'DayType': day_type,
                'Season': season,
                'Consumption_kWh': kwh,
                'Rate_per_kWh': rate,
                'Cost': cost
            })
    
    return pd.DataFrame(results)

# ============================================================================
# BATTERY ARBITRAGE SIMULATION (Simple)
# ============================================================================

def simulate_battery_arbitrage(annual_df, config):
    """
    Simple battery arbitrage: charge at cheapest hour, discharge at most expensive hour
    
    Args:
        annual_df: 8760-row dataframe with hourly rates and consumption
        config: Battery configuration dictionary
    
    Returns:
        Updated dataframe with battery operation and new costs
    """
    df = annual_df.copy()
    
    # Battery parameters
    capacity = config['battery_capacity_kwh']
    efficiency = config['battery_efficiency']
    max_charge = config['max_charge_rate_kw']
    max_discharge = config['max_discharge_rate_kw']
    
    # Add battery columns
    df['Battery_Charge_kWh'] = 0.0
    df['Battery_Discharge_kWh'] = 0.0
    df['Battery_SOC_kWh'] = 0.0
    df['Grid_Import_kWh'] = df['Consumption_kWh'].copy()
    df['Cost_With_Battery'] = df['Cost'].copy()
    df['Arbitrage_Savings'] = 0.0
    
    # Process each day independently for simple arbitrage
    for date in df['Date'].unique():
        day_mask = df['Date'] == date
        day_data = df[day_mask].copy()
        
        # Find cheapest and most expensive hours
        cheapest_idx = day_data['Rate_per_kWh'].idxmin()
        expensive_idx = day_data['Rate_per_kWh'].idxmax()
        
        # Only arbitrage if there's a price difference
        if day_data.loc[expensive_idx, 'Rate_per_kWh'] > day_data.loc[cheapest_idx, 'Rate_per_kWh']:
            
            # Charge at cheapest hour
            charge_amount = min(capacity, max_charge)
            df.loc[cheapest_idx, 'Battery_Charge_kWh'] = charge_amount
            df.loc[cheapest_idx, 'Grid_Import_kWh'] += charge_amount
            df.loc[cheapest_idx, 'Cost_With_Battery'] += charge_amount * day_data.loc[cheapest_idx, 'Rate_per_kWh']
            
            # Discharge at most expensive hour (accounting for efficiency)
            discharge_amount = min(charge_amount * efficiency, max_discharge)
            df.loc[expensive_idx, 'Battery_Discharge_kWh'] = discharge_amount
            df.loc[expensive_idx, 'Grid_Import_kWh'] -= discharge_amount
            df.loc[expensive_idx, 'Cost_With_Battery'] -= discharge_amount * day_data.loc[expensive_idx, 'Rate_per_kWh']
            
            # Calculate savings
            cost_to_charge = charge_amount * day_data.loc[cheapest_idx, 'Rate_per_kWh']
            value_of_discharge = discharge_amount * day_data.loc[expensive_idx, 'Rate_per_kWh']
            df.loc[expensive_idx, 'Arbitrage_Savings'] = value_of_discharge - cost_to_charge
    
    return df

# ============================================================================
# ROI ANALYSIS
# ============================================================================

def calculate_roi(annual_df_with_battery, config):
    """Calculate ROI metrics for battery investment"""
    
    # Annual costs
    baseline_cost = annual_df_with_battery['Cost'].sum()
    cost_with_battery = annual_df_with_battery['Cost_With_Battery'].sum()
    annual_savings = baseline_cost - cost_with_battery
    
    # Investment costs
    battery_cost = config['battery_cost_usd']
    annual_maintenance = battery_cost * config['annual_maintenance_pct']
    lifetime = config['battery_lifetime_years']
    discount_rate = config['discount_rate']
    
    # Simple payback
    simple_payback = battery_cost / annual_savings if annual_savings > 0 else float('inf')
    
    # NPV calculation
    cash_flows = [-battery_cost]  # Initial investment
    for year in range(1, int(lifetime) + 1):
        net_benefit = annual_savings - annual_maintenance
        discounted = net_benefit / ((1 + discount_rate) ** year)
        cash_flows.append(discounted)
    
    npv = sum(cash_flows)
    
    # IRR approximation (simple)
    irr = (annual_savings - annual_maintenance) / battery_cost if battery_cost > 0 else 0
    
    roi_metrics = {
        'Baseline Annual Cost': baseline_cost,
        'Annual Cost With Battery': cost_with_battery,
        'Annual Savings': annual_savings,
        'Battery Investment': battery_cost,
        'Annual Maintenance': annual_maintenance,
        'Simple Payback (years)': simple_payback,
        'NPV': npv,
        'Approximate IRR': irr,
        'Lifetime': lifetime
    }
    
    return roi_metrics

# ============================================================================
# REPORTING & VISUALIZATION
# ============================================================================

def generate_reports(annual_df, roi_metrics, output_prefix='analysis'):
    """Generate summary reports and visualizations"""
    
    # Monthly summary
    monthly = annual_df.groupby(annual_df['Date'].dt.to_period('M')).agg({
        'Consumption_kWh': 'sum',
        'Cost': 'sum',
        'Cost_With_Battery': 'sum',
        'Arbitrage_Savings': 'sum'
    }).reset_index()
    monthly['Month'] = monthly['Date'].astype(str)
    monthly.to_csv(f'{output_prefix}_monthly_summary.csv', index=False)
    
    # Daily summary
    daily = annual_df.groupby('Date').agg({
        'Consumption_kWh': 'sum',
        'Cost': 'sum',
        'Cost_With_Battery': 'sum',
        'Arbitrage_Savings': 'sum'
    }).reset_index()
    daily.to_csv(f'{output_prefix}_daily_summary.csv', index=False)
    
    # Full 8760-hour data
    annual_df.to_csv(f'{output_prefix}_hourly_8760.csv', index=False)
    
    # ROI Report
    with open(f'{output_prefix}_roi_report.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BATTERY STORAGE ROI ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        for key, value in roi_metrics.items():
            if isinstance(value, float):
                f.write(f"{key:.<45} ${value:,.2f}\n")
            else:
                f.write(f"{key:.<45} {value}\n")
    
    print(f"\n✓ Reports generated:")
    print(f"  - {output_prefix}_monthly_summary.csv")
    print(f"  - {output_prefix}_daily_summary.csv")
    print(f"  - {output_prefix}_hourly_8760.csv")
    print(f"  - {output_prefix}_roi_report.txt")
    
    # Visualizations
    create_charts(annual_df, monthly, roi_metrics, output_prefix)

def create_charts(annual_df, monthly_df, roi_metrics, output_prefix):
    """Create visualization charts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Average hourly rate by time of day
    hourly_avg = annual_df.groupby('Hour')['Rate_per_kWh'].mean()
    axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o')
    axes[0, 0].set_title('Average Electricity Rate by Hour of Day')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Rate ($/kWh)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Monthly cost comparison
    axes[0, 1].bar(range(len(monthly_df)), monthly_df['Cost'], alpha=0.7, label='Without Battery')
    axes[0, 1].bar(range(len(monthly_df)), monthly_df['Cost_With_Battery'], alpha=0.7, label='With Battery')
    axes[0, 1].set_title('Monthly Electricity Cost Comparison')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Cost ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative savings over time
    daily_savings = annual_df.groupby('Date')['Arbitrage_Savings'].sum().cumsum()
    axes[1, 0].plot(daily_savings.values)
    axes[1, 0].axhline(y=roi_metrics['Battery Investment'], color='r', linestyle='--', label='Investment Cost')
    axes[1, 0].set_title('Cumulative Battery Arbitrage Savings')
    axes[1, 0].set_xlabel('Day of Year')
    axes[1, 0].set_ylabel('Cumulative Savings ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ROI Summary
    axes[1, 1].axis('off')
    roi_text = f"""
    ROI SUMMARY
    {'='*40}
    
    Annual Savings: ${roi_metrics['Annual Savings']:,.2f}
    Battery Investment: ${roi_metrics['Battery Investment']:,.2f}
    Simple Payback: {roi_metrics['Simple Payback (years)']:.1f} years
    NPV: ${roi_metrics['NPV']:,.2f}
    
    {'='*40}
    """
    axes[1, 1].text(0.1, 0.5, roi_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_charts.png', dpi=300, bbox_inches='tight')
    print(f"  - {output_prefix}_charts.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 60)
    print("ELECTRICITY COST & BATTERY ROI ANALYSIS TOOL")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading battery configuration...")
    config = load_config()
    
    update = input("\nUpdate battery settings? (y/n): ").strip().lower()
    if update == 'y':
        config = update_config()
    
    # Load data
    print("\n2. Loading data files...")
    try:
        profiles = load_profiles('profiles.csv')
        calendar = load_calendar('calendar.csv')
        consumption = load_consumption('consumption_hourly.csv')
        print(f"   ✓ Loaded {len(profiles)} rate entries")
        print(f"   ✓ Loaded {len(calendar)} calendar days")
        print(f"   ✓ Loaded {len(consumption)} hourly consumption values")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure these files exist in the current directory:")
        print("  - profiles.csv")
        print("  - calendar.csv")
        print("  - consumption_hourly.csv")
        return
    
    # Select profile
    print("\n3. Select electricity rate profile:")
    contracts = profiles['Contract'].unique()
    for i, c in enumerate(contracts, 1):
        print(f"   {i}. {c}")
    
    contract_idx = int(input("\nEnter contract number: ")) - 1
    profile_contract = contracts[contract_idx]
    
    # Get available types and options for this contract
    profile_subset = profiles[profiles['Contract'] == profile_contract]
    types = profile_subset['Type'].unique()
    print(f"\nAvailable types: {', '.join(types)}")
    profile_type = input("Enter type: ").strip()
    
    options = profile_subset[profile_subset['Type'] == profile_type]['Option'].unique()
    print(f"Available options: {', '.join(map(str, options))}")
    profile_option = input("Enter option: ").strip()
    
    # Calculate annual cost
    print("\n4. Calculating annual electricity costs...")
    annual_df = calculate_annual_cost(
        profile_contract, profile_type, profile_option,
        profiles, calendar, consumption
    )
    print(f"   ✓ Calculated 8,760 hourly costs")
    
    # Simulate battery
    print("\n5. Simulating battery arbitrage...")
    annual_df_battery = simulate_battery_arbitrage(annual_df, config)
    print(f"   ✓ Battery simulation complete")
    
    # Calculate ROI
    print("\n6. Calculating ROI metrics...")
    roi_metrics = calculate_roi(annual_df_battery, config)
    
    # Generate reports
    print("\n7. Generating reports and charts...")
    generate_reports(annual_df_battery, roi_metrics)
    
    # Display summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nBaseline Annual Cost:        ${roi_metrics['Baseline Annual Cost']:,.2f}")
    print(f"Cost With Battery:           ${roi_metrics['Annual Cost With Battery']:,.2f}")
    print(f"Annual Savings:              ${roi_metrics['Annual Savings']:,.2f}")
    print(f"Simple Payback Period:       {roi_metrics['Simple Payback (years)']:.1f} years")
    print(f"Net Present Value (NPV):     ${roi_metrics['NPV']:,.2f}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()