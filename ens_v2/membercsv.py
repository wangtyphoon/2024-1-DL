import pandas as pd
import os
def member_to_csv(track_data):
    df_new= pd.DataFrame(track_data)
    output_path = f"../route/20170909/csv/group2/{track_data['member']}.csv"
    # Check if the file already exists
    if os.path.exists(output_path):
        # Read existing CSV file
        df_existing = pd.read_csv(output_path)
        
        # Concatenate the new data with the existing data
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Remove duplicate rows if needed (optional, based on all columns)
        df_combined = df_combined.drop_duplicates()
    else:
        # If file does not exist, use the new DataFrame as the combined result
        df_combined = df_new

    # Write combined DataFrame to CSV
    df_combined.to_csv(output_path, index=False)
    print(f"Track data has been saved to {output_path}")

