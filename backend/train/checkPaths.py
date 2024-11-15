import os
from torchvision.io import read_video

def check_paths(base_path):
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return  # Exit the function if the base path is invalid
    else:
        print(f"Base path is valid: {base_path}")
    
    # Iterate through the directories and files in the base path
    for root, dirs, files in os.walk(base_path):
        for name in dirs:
            dir_path = os.path.join(root, name)
            #print(f"Directory exists: {dir_path}")
        
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.isfile(file_path):
               # print(f"File exists: {file_path}")
                
                # Optional: Check if the file is a valid video
                try:
                    video, _, _ = read_video(file_path, pts_unit='sec')
                except Exception as e:
                    print(f"Cannot read video {file_path}: {e}")
            else:
                print(f"Not a valid file: {file_path}")

if __name__ == "__main__":
    base_path = "/work/cssema416/202510/05/05/data/kinetics-dataset/k700-2020/train"  # Update this to your actual dataset path
    
    print("Starting file and directory check...")
    check_paths(base_path)
    print("Check completed.")
    