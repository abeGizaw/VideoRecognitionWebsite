/**
 * Import function triggers from their respective submodules:
 *
 * import {onCall} from "firebase-functions/v2/https";
 * import {onDocumentWritten} from "firebase-functions/v2/firestore";
 *
 * See a full list of supported triggers at https://firebase.google.com/docs/functions
 */

// import {onRequest} from "firebase-functions/v2/https";
// import * as logger from "firebase-functions/logger";
// functions/src/index.ts

// import * as functions from 'firebase-functions';
// import * as axios from 'axios'; // To send requests to your Flask app

// // This function will proxy video upload requests to your Flask server
// export const proxyVideoUpload = functions.https.onRequest(async (req, res) => {
//   try {
//     const apiUrl = 'http://your-flask-app-url/upload'; // Your Flask API URL
    
//     // Forward request to Flask API
//     const response = await axios.post(apiUrl, req.body, {
//       headers: req.headers,
//     });
    
//     res.status(response.status).send(response.data);
//   } catch (error) {
//     console.error('Error uploading video', error);
//     res.status(500).send('Failed to upload video');
//   }
// });
// Start writing functions
// https://firebase.google.com/docs/functions/typescript

// export const helloWorld = onRequest((request, response) => {
//   logger.info("Hello logs!", {structuredData: true});
//   response.send("Hello from Firebase!");
// });
