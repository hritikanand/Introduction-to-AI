import pandas as pd
from geopy.distance import geodesic
from itertools import combinations

# === STEP 1: Load SCATS Excel 
df = pd.read_excel("Scats Data October 2006.xls", sheet_name='Data', engine='xlrd')

# First row is header info, so reassign columns
df.columns = df.iloc[0]
df = df.drop(index=0)

# === STEP 2: Extract SCATS ID + Location + Latitude + Longitude ===
df_cleaned = df[['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates()
df_cleaned.columns = ['SCATS_ID', 'Location', 'Latitude', 'Longitude']

# Ensure numeric types for coordinates
df_cleaned['Latitude'] = df_cleaned['Latitude'].astype(float)
df_cleaned['Longitude'] = df_cleaned['Longitude'].astype(float)

# Save cleaned site data for future GUI or mapping use
df_cleaned.to_csv("scats_locations.csv", index=False)
print(f"scats_locations.csv created with {len(df_cleaned)} unique SCATS sites.")

# === STEP 3: Generate Distance Matrix ===
records = []
scats_pairs = combinations(df_cleaned.itertuples(index=False), 2)

for a, b in scats_pairs:
    coord_a = (a.Latitude, a.Longitude)
    coord_b = (b.Latitude, b.Longitude)
    distance_km = geodesic(coord_a, coord_b).km
    records.append({'From_SCATS': a.SCATS_ID, 'To_SCATS': b.SCATS_ID, 'Distance_km': distance_km})
    records.append({'From_SCATS': b.SCATS_ID, 'To_SCATS': a.SCATS_ID, 'Distance_km': distance_km})  # symmetric

# Save distance matrix
distance_df = pd.DataFrame(records)
distance_df.to_csv("distance_matrix.csv", index=False)
print(f"distance_matrix.csv created with {len(distance_df)} entries.")
