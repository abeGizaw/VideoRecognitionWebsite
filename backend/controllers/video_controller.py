import traceback
from flask import request, jsonify
from models.video_model import process_video 
from utils.index import save_video_to_dir

# Controller method to handle video upload and processing
def upload_and_process_video():
    try: 
        if 'file' not in request.files:
            return jsonify({'message': 'No file uploaded'}), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400

        # Save the file using helper function
        file_path = save_video_to_dir(file)

        # Process the video with your CNN model
        result = process_video(file_path)

        return jsonify({'message': f'Video {file.filename} processed successfully. {result}'})
    except Exception as e:
        # Log the exception for debugging
        print("Error in upload_and_process_video:", traceback.format_exc())

        # Return a JSON response with an error message and appropriate status code
        return jsonify({'error': 'An error occurred during video processing.'}), 500
