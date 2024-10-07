import { Box, Button, Typography } from '@mui/joy';
import { useState } from 'react';
export const Page = () => {
  const [message, setMessage] = useState<string>('');
  const [videoFile, setVideoFile] = useState<File | null>(null);

  const handleUpload = async () => {
    console.log(videoFile);
    if (videoFile) {
      try {
        // Create FormData and append the video file
        const formData = new FormData();
        formData.append('file', videoFile);

        setMessage('Uploading file...');

        // Send the file to the backend
        // const uploadResponse = await fetch('http://127.0.0.1:5001/upload', {
        //   method: 'POST',
        //   body: formData,
        // });
        const uploadResponse = await fetch(
          `${import.meta.env.VITE_BACKEND_URL}/upload`,
          {
            method: 'POST',
            body: formData,
            headers: {
              Accept: 'application/json',
            },
          },
        );

        const result = await uploadResponse.json();
        setMessage(result.message);
      } catch (error) {
        console.error('Error uploading file:', error);
        setMessage('Error uploading file');
      }
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
      setMessage(`Dropped video: ${file.name}`);
    } else {
      setMessage('Please drop a valid video file.');
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
      setMessage(`Selected video: ${file.name}`);
    } else {
      setMessage('Please upload a valid video file.');
    }
  };

  return (
    <Box
      sx={{
        width: 'fit-content',
        margin: 'auto',
        height: '100vh',
        alignItems: 'center',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
      }}
    >
      <Typography level='h1'>Upload Video</Typography>
      {/* Drag and Drop Area */}
      <Box
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        sx={{
          border: '2px dashed gray',
          padding: '2rem',
          textAlign: 'center',
          width: '300px',
          borderRadius: '10px',
          backgroundColor: '#f9f9f9',
          cursor: 'pointer',
        }}
      >
        <Typography>Drag and drop a video here</Typography>
      </Box>

      <input
        type='file'
        accept='video/*'
        style={{ display: 'none' }}
        id='video-upload'
        onChange={handleFileChange}
      />
      <label htmlFor='video-upload'>
        <Button component='span'>Choose a Video</Button>
      </label>

      <Button onClick={handleUpload}>Upload</Button>
      {message && <Typography>{message}</Typography>}
    </Box>
  );
};
