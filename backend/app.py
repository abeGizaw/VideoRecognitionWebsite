from flask import Flask
from flask_cors import CORS
from controllers.video_controller import upload_and_process_video
import os

# Initialize the Flask app
app = Flask(__name__)
# CORS(app, resources={r"/upload": {"origins": "https://what-tha-vid-do.web.app"}})
CORS(app)

# Set a folder for temporary uploads
UPLOAD_FOLDER = '/tmp/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_video():
    return upload_and_process_video()

@app.route('/')
def index():
    return "Flask server is running!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Use the port from environment, default to 8080
    app.run(host='0.0.0.0', port=port)
