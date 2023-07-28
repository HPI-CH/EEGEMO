import os
import pandas as pd

def combine_csv_files(directory_path):
    # Get a list of all .csv files in the directory
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]
    
    # If there are no .csv files in the directory, print a message and return
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    # Initialize an empty DataFrame to hold the merged data
    merged_data = pd.DataFrame()
    
    # Read and merge the .csv files with the same column structure
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        data = pd.read_csv(file_path)
        
        # Check if the columns match with the existing DataFrame
        if not merged_data.empty and list(data.columns) != list(merged_data.columns):
            print(f"Columns in file '{file}' do not match the existing DataFrame. Skipping this file.")
            continue
        
        # Concatenate the data to the existing DataFrame
        merged_data = pd.concat([merged_data, data], ignore_index=True)
        
    # If there is no data to merge, print a message and return
    if merged_data.empty:
        print("No data to merge.")
        return
    
    # Save the merged DataFrame to a new CSV file in the same directory
    merged_file_path = os.path.join(directory_path, 'merged_data.csv')
    merged_data.to_csv(merged_file_path, index=False)
    
    print(f"Merged data saved to {merged_file_path}")

if __name__ == "__main__":
    # Provide the directory path as input
    directory_path = input("Enter the directory path: ")
    combine_csv_files(directory_path)
