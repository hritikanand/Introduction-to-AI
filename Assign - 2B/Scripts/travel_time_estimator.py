import pandas as pd
import math

distance_df = pd.read_csv("distance_matrix.csv")
traffic_flow_df = pd.read_csv("SCATS_timeseries.csv")

# Constants
delay_seconds = 30
delay_hours = delay_seconds / 3600   
speed_limit_kmh = 60  # Max speed cap

# Function
def flow_to_speed(flow):
    # Quadratic: flow = -1.4648375 * speed^2 + 93.75 * speed
    a = -1.4648375
    b = 93.75
    c = -flow

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return 0  
 
    root1 = (-b + math.sqrt(discriminant)) / (2 * a)
    root2 = (-b - math.sqrt(discriminant)) / (2 * a)
    speed = min(root1, root2) 
 
    return min(speed, speed_limit_kmh)


estimated_travel_times = []  
 
for _, row in distance_df.iterrows():
    from_scats = row['From_SCATS']
    to_scats = row['To_SCATS']
    distance_km = row['Distance_km']
    volume = traffic_flow_df[traffic_flow_df['SCATS_ID'] == from_scats]['Volume'].mean()

    if pd.isna(volume):
        volume = 0
 
    speed_kmh = flow_to_speed(volume)  
    if speed_kmh <= 0:
        travel_time_hours = None
    else:
  
        travel_time_hours = (distance_km / speed_kmh) + delay_hours

    # Store the result
    estimated_travel_times.append({
        'From_SCATS': from_scats,
        'To_SCATS': to_scats,
        'Distance_km': distance_km,
        'Estimated_Speed_kmh': speed_kmh,
        'Estimated_Travel_Time_Hours': travel_time_hours,
        'Traffic_Flow_Volume': volume
    })


estimated_travel_times_df = pd.DataFrame(estimated_travel_times)
estimated_travel_times_df.to_csv("estimated_travel_times.csv", index=False) 

# Output info
print("estimated_travel_times.csv created with:", estimated_travel_times_df.shape)  


