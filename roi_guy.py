import tkinter as tk
from tkinter import messagebox

from electricity_roi import (
    setup_directories,
    load_config,
    load_profiles,
    load_calendar,
    load_consumption,
    calculate_annual_cost,
    simulate_battery_arbitrage,
    generate_monthly_summary,
    calculate_10year_roi,
    generate_reports,
    DEFAULT_BATTERY_CONFIG,
    DEFAULT_INSTALLATION_CONFIG
)

def run_analysis():
    try:
        setup_directories()
        battery_config = load_config('battery_config.json', DEFAULT_BATTERY_CONFIG)
        installation_config = load_config('installation_cost.json', DEFAULT_INSTALLATION_CONFIG)
        profiles = load_profiles()
        calendar = load_calendar()
        consumption = load_consumption()

        profile_contract = 'YourContract'
        profile_type = 'TOU'
        profile_option = '1'

        annual_df = calculate_annual_cost(profile_contract, profile_type, profile_option, profiles, calendar, consumption)
        annual_df_battery = simulate_battery_arbitrage(annual_df, battery_config)
        monthly_summary = generate_monthly_summary(annual_df_battery)
        annual_savings = monthly_summary[monthly_summary['Month'] == 13]['Savings'].values[0]
        roi_projection, roi_metrics = calculate_10year_roi(annual_savings, battery_config, installation_config)

        generate_reports(annual_df_battery, monthly_summary, roi_projection, roi_metrics, profile_contract)
        messagebox.showinfo("Success", "Analysis complete. Reports saved.")
    except Exception as e:
        print(f"Error: {e}")  # ✅ This prints to Command Prompt
        messagebox.showerror("Error", str(e))  # This shows the popup

import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.title("Electricity ROI Tool")

tk.Label(root, text="Click to run full ROI analysis").pack(pady=10)
tk.Button(root, text="Run ROI Analysis", command=run_analysis).pack(pady=20)

root.mainloop()