from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from controllers.video_controller import upload_and_process_video


# Intialize the Flask app
app = Flask(__name__)
CORS(app)  

# Set a folder for temporary uploads
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_video():
    return upload_and_process_video()
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
