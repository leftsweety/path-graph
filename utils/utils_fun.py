import os
import sys
import shutil

def list_files(directory):
    files = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                files.append(entry.name)
            if entry.is_dir():
                files.append(entry.name)
    return files

def generate_repeated_list(value, length):
    return [value] * length

# Store the original sys.path
original_sys_path = sys.path.copy()
# Function to reset sys.path
def reset_sys_path():
    global sys
    sys.path = original_sys_path.copy()

def delete_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' has been deleted.")
    else:
        print(f"Directory '{directory_path}' does not exist.")

def ensure_dir(directory):
    """
    Check if a directory exists, and if not, create it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

def check_file_exist(file_name):
    if os.path.exists(file_name):
        return True
    else:
        return False