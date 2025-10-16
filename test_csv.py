import pandas as pd

calendar = pd.read_csv('calendar.csv')
print("Columns found:", list(calendar.columns))
print("\nFirst 3 rows:")
print(calendar.head(3))