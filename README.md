# Video Recognition Website
[Webstie](https://what-tha-vid-do.web.app/)

Currently, there is no working deployed version of this website due to the large storage requirements, which are challenging to manage on Google Cloud. To use the website, you will need to run it locally.  
This project allows users to upload or record videos for action classification using a Swin3db model. To run this website locally, follow the steps below.


## Getting the website Started
I recommend running this on a server with GPUs (Example Rose Server -- uses Linux)
### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/abeGizaw/VideoRecognitionWebsite.git
```

### 2. **Navigate to Project Directory**
After cloning the repository, navigate into the project directory:
```bash
cd VideoRecognitionWebsite
```

### 3. **Setup Conda Environment**
To set up the environment for this project, follow these steps:

1. **Create and activate a new Conda environment:**

    ```bash
    conda create -n video_env python=3.12 -y
    conda activate video_env
    ```

2. **Install PyTorch with CUDA, torchvision, and torchaudio. This assumes you are using linux:**
- For Linux
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
    ```
- For Mac
    ```bash
    conda install pytorch::pytorch torchvision torchaudio -c pytorch
    ```
- For Windows
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```

3. **Install additional libraries:**

    ```bash
    conda install -c conda-forge matplotlib -y
    conda install -c conda-forge av -y
    conda install -c conda-forge joblib -y
    conda install -c conda-forge flask-cors
    conda install -c conda-forge flask-cors
    conda install flask
    pip install transformers
    pip install accelerate
    pip install sentencepiece
    ```

This setup will install all the necessary dependencies for running the program, including PyTorch, CUDA support, and essential libraries for data processing and visualization.


### 4. **Start the Backend (Flask)**
   - Change into the backend directory:
     ```bash
     cd backend
     ```
   - Run the Flask application:
     ```bash
     python app.py
     ```

### 5. **Start the Frontend (React)**
   - Open a new terminal window.
   - Navigate back to the root directory:
     ```bash
     cd ..
     ```
   - Change into the frontend directory:
     ```bash
     cd frontend
     ```
   - Start the frontend development server:
     ```bash
     npm install
     npm run dev
     ```

This should start the website on localhost:3000.

## Usage
Now that both the backend and frontend are running, you can open your browser and go to http://localhost:3000 to use the website.

# Project Structure and Key Files

This section provides an overview of the key directories and files within the project, focusing on the main components that facilitate the functionality of this application.


## Key Directories and Files

### 1. `backend/`

- **`app.py`**  
  This file serves as the primary entry point for the backend of the application. It initializes the Flask application, sets up essential routes, and configures middleware, including CORS for handling cross-origin requests. The `app.py` file orchestrates the communication between the frontend and backend, managing HTTP requests and responses. It’s responsible for invoking appropriate backend functionality, including model processing and video handling.

### 2. `backend/models/`

- **`video_model.py`**  
  This file loads the pre-trained video recognition model and handles video processing sent from the website. It utilizes the specified model architecture and pre-trained weights to analyze video inputs. The `video_model.py` file returns a JSON response with the top 5 predicted categories, offering confidence scores for each. It leverages PyTorch to perform inference and prepares the output for the frontend to display results.

- **`chatbot_model.py`**  
  Code for Chatbot 

### 3. `backend/train/`

- **`index.py`**  
  This file is the main training script that drives the model training process. It loads the Kinetics dataset, modifies the Swin model to include new classes, and performs training and evaluation on the GPU. `index.py` uses helper functions from `trainHelper.py` and saves the trained model weights after each session. This file is designed for extensive model training workflows with multiple classes.

- **`base.py`**  
  This file is similar to `index.py` but with a key difference: it only adds a single new class to the original Swin model's classes, creating a base model to serve as a fallback. It also saves its own set of weights separately, ensuring that the model can revert to this configuration if needed.

- **`trainMappings.py`**  
  Contains mappings of class labels, including the label-to-index mappings for Kinetics-400 classes and any additional custom classes. This file also provides a function to generalize similar classes, which helps consolidate class groups when appropriate.

- **`videoCreator.py`**  
  This file is responsible for loading and preprocessing video data for model training. It includes the custom DataLoader code that prepares video data for the model, along with error-checking routines for handling corrupted videos or files with unexpected formats. 

- **`trainHelper.py`**  
  A utility file that provides helper functions supporting the training workflow. It includes functions for caching label mappings, generating dataframes for testing, and managing common tasks like logging and checkpoint creation. These utilities make the primary training scripts more modular and maintainable.

- **`baseline.py`**
  Baseline Code  

---

This structure provides an overview of the primary components within the project. Each section can be expanded upon to explain specific functions, classes, or processes as needed.

## How to train and test model
Simply run 
```bash
  python backend/train/index.py
 ```
within the VideoRecogniton Folder. **MAKE SURE YOU ARE ON A BRANCH WHEN DOING THIS**

## Notes and Warnings
Ensure you have all necessary dependencies installed (pip for Python dependencies and npm for frontend dependencies).  
If you encounter any issues, make sure you are using compatible versions of Python and Node.js as specified in the repository requirements.  
When running the website, the first time you upload a video, the model weights will be uploaded locally. The model is about 364 MB. After it downloads once, the website will run a lot faster. 
