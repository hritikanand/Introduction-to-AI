import pandas as pd
from datetime import datetime, timedelta

# Define time mapping function
def get_time_from_vcode(vcode):
    index = int(vcode[1:])
    return timedelta(minutes=15 * index)

# Load Excel
file_path = "Scats Data October 2006.xls"
df = pd.read_excel(file_path, sheet_name="Data", engine="xlrd")

# Clean header
df.columns = df.iloc[0]
df = df.drop(index=0).reset_index(drop=True)
df = df.rename(columns={df.columns[0]: 'SCATS_ID', df.columns[1]: 'Location'})
df = df.loc[:, ~df.columns.duplicated()]
df.columns = [str(c) for c in df.columns]

# Melt to long format
melt_df = df.melt(id_vars=['SCATS_ID', 'Location', 'Date'], var_name='TimeCode', value_name='Volume')

# Only keep rows with valid V-codes
valid_vcodes = [f"V{str(i).zfill(2)}" for i in range(96)]
melt_df = melt_df[melt_df['TimeCode'].isin(valid_vcodes)]

# Filter rows with clean date format
melt_df = melt_df[pd.to_datetime(melt_df['Date'], errors='coerce').notna()]
melt_df['Date'] = pd.to_datetime(melt_df['Date'], errors='coerce')

# Create timestamp
melt_df['Timestamp'] = melt_df['Date'] + melt_df['TimeCode'].apply(get_time_from_vcode)

# Final cleanup
melt_df = melt_df.drop(columns=['Date', 'TimeCode'])
melt_df = melt_df[['Timestamp', 'SCATS_ID', 'Location', 'Volume']]
melt_df['Volume'] = pd.to_numeric(melt_df['Volume'], errors='coerce')
melt_df = melt_df.dropna(subset=['Volume'])
melt_df['Volume'] = melt_df['Volume'].astype(int)

# Save
melt_df.to_csv("SCATS_timeseries.csv", index=False)
print("✅ SCATS_timeseries.csv created with:", melt_df.shape)
