import os
import glob
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def process_image(img_path):
    try:
        with Image.open(img_path) as img:
            if img.size == (448, 256):
                return f"Skip {img_path}"

            img_resized = img.resize((448, 256))
            img_resized.save(img_path)
    except Exception as e:
        return f"Error processing {img_path}: {e}"

def process_sub_sub_dir(sub_sub_dir_path):
    print(f"Processing {sub_sub_dir_path}")
    img_paths = glob.glob(os.path.join(sub_sub_dir_path, '*.png'))

    with ThreadPoolExecutor() as executor:
        for result in executor.map(process_image, img_paths):
            if result:
                print(result)

# Define the root directory
root_dir = '/scratch/zczqyc4/360-videos-grouped/'

# Iterate through all subdirectories under the root
for sub_dir in os.listdir(root_dir):

    sub_dir_path = os.path.join(root_dir, sub_dir)

    # Check if it's a directory
    if os.path.isdir(sub_dir_path):
        print(f"start processing {sub_dir_path}")

        # Iterate through all sub-subdirectories under sub_dir
        for sub_sub_dir in os.listdir(sub_dir_path):
            sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)

            # Check if it's a directory
            if os.path.isdir(sub_sub_dir_path):
                process_sub_sub_dir(sub_sub_dir_path)
