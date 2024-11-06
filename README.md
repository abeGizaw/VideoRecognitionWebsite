# Video Recognition Website
[Webstie](https://what-tha-vid-do.web.app/)

Currently, there is no working deployed version of this website due to the large storage requirements, which are challenging to manage on Google Cloud. To use the website, you will need to run it locally.  
This project allows users to upload or record videos for action classification using a Swin3db model. To run this website locally, follow the steps below.

## Getting Started

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/abeGizaw/VideoRecognitionWebsite.git
```

### 2. **Checkout the Local Development Branch**
After cloning the repository, navigate into the project directory and switch to the `local_dev` branch:
```bash
git checkout local_decd VideoRecognitionWebsite
git checkout local_dev
```

### 3. **Start the Backend (Flask)**
   - Change into the backend directory:
     ```bash
     cd backend
     ```
   - Run the Flask application:
     ```bash
     python app.py
     ```

### 4. **Start the Frontend (React)**
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
     npm run dev
     ```

This should start the website on localhost:3000.

## Usage
Now that both the backend and frontend are running, you can open your browser and go to http://localhost:3000 to use the website.

## Notes
Ensure you have all necessary dependencies installed (pip for Python dependencies and npm for frontend dependencies).
If you encounter any issues, make sure you are using compatible versions of Python and Node.js as specified in the repository requirements.
