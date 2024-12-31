import traceback
from flask import request, jsonify
from models.video_model import process_video
from utils.index import save_video_to_dir
from models.chatbot_model import small_chatbot_model, large_chatbot_model, aiSuite_chatbot_model

# Controller method to handle video upload and processing
def upload_and_process_video(source: str) -> str:
    try: 
        if 'file' not in request.files:
            return jsonify({'message': 'No file uploaded'}), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400

        # Save the file using helper function
        file_path = save_video_to_dir(file)

        # Process the video with out model
        result, mostConfident = process_video(file_path)

        if source == 'chatBot':
            # result = aiSuite_chatbot_model(result)
            reuslt = small_chatbot_model(mostConfident)
            # result = large_chatbot_model(result)

        return jsonify({'message': f'{result}'})
    except Exception as e:
        # Log the exception for debugging
        print("Error in upload_and_process_video:", traceback.format_exc())

        return jsonify({'error': 'An error occurred during video processing.'}), 500
