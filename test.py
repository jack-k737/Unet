from glob import glob
import os

image_dir_mode = os.path.join(os.getcwd(), 'ODOC','*','*','*','*.npy')
image_dirs = glob(image_dir_mode)
print(image_dirs)

for file_path in image_dirs:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")