// Might not need this file
const express = require('express');
const cors = require('cors');
const app = express();
const port = 5000;

app.use(cors());   

app.post('/upload', (req, res) => {
    console.log('File uploaded');
    res.json({ message: 'A button got clicked' });
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
  });