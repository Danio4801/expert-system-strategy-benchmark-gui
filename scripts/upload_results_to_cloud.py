import os
import glob
from integrations.firebase_client import FirebaseConnector


KEY_FILE = "projektsystemekspertowy-firebase-adminsdk-fbsvc-c6d7fc4e7f.json"
CSV_PATH = "results/final_clustering_benchmark.csv"
VISUALIZATIONS_DIR = "results/visualizations"

def main():
    print("Starting Cloud Upload Process...")
    

    try:
        connector = FirebaseConnector(KEY_FILE)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not initialize Firebase connection.\n{e}")
        return


    try:
        if os.path.exists(CSV_PATH):
            connector.upload_experiments(CSV_PATH)
        else:
            print(f"Error: CSV file not found at {CSV_PATH}")
    except Exception as e:
        print(f"Error uploading CSV: {e}")


    try:

        png_files = glob.glob(os.path.join(VISUALIZATIONS_DIR, "*.png"))
        
        if not png_files:
            print(f"No PNG files found in {VISUALIZATIONS_DIR}")
        
        for file_path in png_files:
            filename = os.path.basename(file_path)

            destination = f"visualizations/{filename}"
            
            try:
                url = connector.upload_image(file_path, destination)
                if url:
                    print(f"Uploaded {filename} -> {url}")
            except Exception as e:
                print(f"Failed to upload {filename}: {e}")
                
    except Exception as e:
        print(f"Error processing visualizations: {e}")

    print("Cloud Upload Process Completed.")

if __name__ == "__main__":
    main()
