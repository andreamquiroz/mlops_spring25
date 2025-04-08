import pandas as pd

# Load the dataset from the data folder
cars_df = pd.read_csv("data/CAR DETAILS FROM CAR DEKHO.csv")

# Add a new column converting km_driven to mi_driven
cars_df['mi_driven'] = cars_df['km_driven'] * 0.621371

# Save the processed DataFrame to a new CSV in the data folder
processed_path = "data/processed_car_details.csv"
cars_df.to_csv(processed_path, index=False)

print(f"Data processing complete. Processed file saved to {processed_path}")