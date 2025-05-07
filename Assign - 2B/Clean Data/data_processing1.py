import pandas as pd
from datetime import datetime, timedelta

# Load SCATS Excel file
file_path = 'Scats Data October 2006.xls'
df = pd.read_excel(file_path, sheet_name='Data', engine='xlrd')

# Set first row as header
df.columns = df.iloc[0]
df = df.drop(index=0).reset_index(drop=True)

# Rename SCATS_ID and Location columns
df.rename(columns={df.columns[0]: 'SCATS_ID', df.columns[1]: 'Location'}, inplace=True)
df = df.loc[:, ~df.columns.isnull()]
df = df.loc[:, ~df.columns.duplicated()]
df.columns = [str(col) if not isinstance(col, str) else col for col in df.columns]

# Filter and melt
valid_vcodes = [f"V{str(i).zfill(2)}" for i in range(96)]
melt_df = df.melt(id_vars=['SCATS_ID', 'Location'], var_name='TimeCode', value_name='Volume')
melt_df = melt_df[melt_df['TimeCode'].isin(valid_vcodes)]

# Sort by SCATS and TimeCode to align with timestamps
melt_df = melt_df.sort_values(['SCATS_ID', 'TimeCode'])

# Build timestamp column
# Get number of unique SCATS IDs
n_sites = df['SCATS_ID'].nunique()
# One row per site per V-code per day → 96 × 31 × n_sites
timestamps = []
for day in range(31):
    date = datetime(2006, 10, 1) + timedelta(days=day)
    for v in range(96):
        timestamps.append(date + timedelta(minutes=15 * v))
timestamps_all = timestamps * n_sites

# Repeat timestamps across all SCATS_IDs properly
timestamps_all = sorted(timestamps_all * (len(melt_df) // len(timestamps)))

# Add Timestamp column
melt_df['Timestamp'] = timestamps_all[:len(melt_df)]

# Clean and finalize
melt_df['Volume'] = pd.to_numeric(melt_df['Volume'], errors='coerce')
melt_df = melt_df.dropna(subset=['Volume'])
melt_df['Volume'] = melt_df['Volume'].astype(int)
melt_df['SCATS_ID'] = melt_df['SCATS_ID'].astype(int).astype(str).str.zfill(4)

# Save final long-format CSV
melt_df = melt_df[['SCATS_ID', 'Location', 'Timestamp', 'Volume']]
melt_df.to_csv('SCATS_timeseries_long.csv', index=False)
print("✅ SCATS_timeseries_long.csv created with", melt_df.shape[0], "rows.")
