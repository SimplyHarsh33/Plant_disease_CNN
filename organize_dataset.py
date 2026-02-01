
import os
import shutil
from pathlib import Path

# Configuration
BASE_DIR = Path(r"d:\Downloads-D\AI-plant_disease")
DATASET_DIR = BASE_DIR / "Dataset"
OLD_DB_DIR = BASE_DIR / "Testing_Database"

# Folders to move
TARGET_FOLDERS = [
    "Apple", "Bell Pepper", "Cherry", "Corn (Maize)", "Grape", 
    "Peach", "Potato", "Raspberry", "Soybean", "Squash", 
    "Strawberry", "Tomato"
]

def organize():
    print(f"[*] Base Directory: {BASE_DIR}")
    
    # 1. Create Dataset Directory
    if not DATASET_DIR.exists():
        try:
            DATASET_DIR.mkdir()
            print(f"[*] Created 'Dataset' directory at {DATASET_DIR}")
        except Exception as e:
            print(f"[!] Error creating directory: {e}")
            return
    else:
        print(f"[*] 'Dataset' directory already exists.")

    # 2. Move Folders
    for folder_name in TARGET_FOLDERS:
        source = BASE_DIR / folder_name
        destination = DATASET_DIR / folder_name
        
        if source.exists():
            try:
                print(f"Moving {folder_name}...", end=" ")
                shutil.move(str(source), str(destination))
                print("DONE")
            except Exception as e:
                print(f"\n[!] Failed to move {folder_name}: {e}")
        else:
            print(f"Skipping {folder_name} (Not found in base dir)")

    # 3. Remove Testing_Database
    if OLD_DB_DIR.exists():
        try:
            print(f"Removing old Testing_Database...", end=" ")
            shutil.rmtree(OLD_DB_DIR)
            print("DONE")
        except Exception as e:
            print(f"\n[!] Failed to remove Testing_Database: {e}")
    else:
        print("[*] Testing_Database not found (already removed?)")

if __name__ == "__main__":
    organize()
