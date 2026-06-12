import os
import glob
from features import generate_features

def process_all_stocks(raw_dir: str, processed_dir: str):
    os.makedirs(processed_dir, exist_ok = True)
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    print(f"Found {len(csv_files)} files.\n")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")

        try:
            processed_df = generate_features(file_path, prediction_horizon = 5)
            save_path = os.path.join(processed_dir, filename)
            processed_df.to_csv(save_path, index=False)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("All files processed successfully!")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../data/raw"))
    PROCESSED_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../data/processed"))
    print(f"Looking for raw data in: {RAW_DATA_PATH}")
    process_all_stocks(RAW_DATA_PATH, PROCESSED_DATA_PATH)
