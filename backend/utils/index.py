# utils/video_processing.py
import os
from flask import current_app

def save_video_to_dir(file):
    """
    Helper function to save the uploaded video to a specified directory.
    """
    upload_folder = current_app.config['UPLOAD_FOLDER']
    
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    
    return file_path
