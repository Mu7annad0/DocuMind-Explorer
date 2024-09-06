import shutil  # Import shutil module for file and directory operations
from pathlib import Path  # Import Path class for file path handling
from typing import List 

from streamlit.runtime.uploaded_file_manager import UploadedFile  # Import UploadedFile class from Streamlit

from rag.config import Config

def upload_file(files: List[UploadedFile], remove_old_files: bool = True) -> List[Path]:
    if remove_old_files:
        # Remove old database and documents directories if remove_old_files is True
        shutil.rmtree(Config.Path.DATABASE_DIR, ignore_errors=True)
        shutil.rmtree(Config.Path.DOCUMENTS_DIR, ignore_errors=True)
    
    # Create the documents directory, including any necessary parent directories
    Config.Path.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    files_paths = []  # Initialize an empty list to store paths of uploaded files
    for file in files:
        # Create a new file path by joining the documents directory path with the uploaded file name
        file_path = Config.Path.DOCUMENTS_DIR / file.name
        
        # Open the file in binary write mode and write the contents of the uploaded file
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        
        files_paths.append(file_path)  # Add the new file path to the list
    
    return files_paths  # Return the list of paths for all uploaded files
