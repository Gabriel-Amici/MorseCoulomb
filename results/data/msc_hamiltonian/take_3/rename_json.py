import os
import json
import glob

def rename_json_files(directory_path):
    """
    Rename JSON files in the specified directory based on properties inside the JSON.
    
    Parameters:
    directory_path (str): Path to the directory containing JSON files
    """
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for file_path in json_files:
        try:
            # Load JSON data from file
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            
            # Extract required values
            try:
                # Extract alpha value and format as a--x-y
                alpha_value = json_data["alpha"]
                alpha_str = f"a--{alpha_value}".replace(".", "-")
                
                # Extract position_grid_extension and format as L--x
                grid_extension = json_data["position_grid"]["position_grid_extension"]
                grid_extension_str = f"L--{int(grid_extension)}"
                
                # Extract position_grid_size and format as N-x
                grid_size = json_data["position_grid"]["position_grid_size"]
                grid_size_str = f"N--{int(grid_size)}"
                
                # Combine all components for the new filename
                directory = os.path.dirname(file_path)
                new_name = f"{alpha_str}--{grid_extension_str}--{grid_size_str}.json"
                new_path = os.path.join(directory, new_name)
                
                # Rename the file
                os.rename(file_path, new_path)
                print(f"Renamed: {os.path.basename(file_path)} → {new_name}")
                
            except KeyError as e:
                print(f"Error: Missing required key {e} in file {file_path}")
                
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON from file {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("Renaming complete")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = input("Enter the directory path containing JSON files: ")
    
    rename_json_files(directory)
