import pandas as pd

consumption = pd.read_csv('consumption_hourly.csv')
print("Columns found:", list(consumption.columns))
print("\nFirst 3 rows:")
print(consumption.head(3))