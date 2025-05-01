import pandas as pd
from datetime import datetime, timedelta

# Load the Excel file and select the correct sheet
scats_file = 'Scats Data October 2006.xls'
df = pd.read_excel(scats_file, sheet_name='Data', engine='xlrd')

# Set first row as header
df.columns = df.iloc[0]
df = df.drop(index=0).reset_index(drop=True)

# Rename important columns
df.rename(columns={df.columns[0]: 'SCATS_ID', df.columns[1]: 'Location'}, inplace=True)

# Clean up: remove null or duplicate columns
df = df.loc[:, ~df.columns.isnull()]
df = df.loc[:, ~df.columns.duplicated()]
df.columns = [str(col) if not isinstance(col, str) else col for col in df.columns]

# Melt wide format to long format (V00–V95)
df_long = df.melt(id_vars=['SCATS_ID', 'Location'], var_name='TimeCode', value_name='Volume')

# Only keep rows where TimeCode is between V00 and V95
valid_vcodes = [f"V{str(i).zfill(2)}" for i in range(96)]
df_long = df_long[df_long['TimeCode'].isin(valid_vcodes)]

# Format SCATS_ID properly (e.g., 970 → 0970)
df_long['SCATS_ID'] = df_long['SCATS_ID'].astype(str).str.zfill(4)

# Map V-codes to actual timestamps (fixed to 1 Oct 2006, 00:00 + 15-min intervals)
def vcode_to_time(code):
    index = int(code[1:])  # remove "V"
    return (datetime(2006, 10, 1) + timedelta(minutes=15 * index)).strftime("%Y-%m-%d %H:%M:%S")

df_long['Timestamp'] = df_long['TimeCode'].apply(vcode_to_time)

# Clean Volume column: convert to numeric, drop missing, cast to int
df_long['Volume'] = pd.to_numeric(df_long['Volume'], errors='coerce')
df_long = df_long.dropna(subset=['Volume'])
df_long['Volume'] = df_long['Volume'].astype(int)

# Final cleanup
df_long = df_long.drop(columns=['TimeCode'])
df_long['Timestamp'] = pd.to_datetime(df_long['Timestamp'])

# Pivot to wide format (Timestamp × SCATS_ID) 
df_pivot = df_long.pivot_table(index='Timestamp', columns='SCATS_ID', values='Volume', aggfunc='first')

# Fill any missing values 
df_pivot = df_pivot.fillna(0).astype(int)

# Save to CSV
df_pivot.to_csv("SCATS_timeseries.csv")
print("SCATS_timeseries.csv created", df_pivot.shape)
