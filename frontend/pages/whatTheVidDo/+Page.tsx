import { Box, Button, Typography } from '@mui/joy';
import { useState } from 'react';
import testVideo from '../../assets/testVid.mov';
export const Page = () => {
  const [message, setMessage] = useState<string>('');

  const handleUpload = async () => {
    try {
      // Fetch the video from the assets folder
      const response = await fetch(testVideo);
      const videoBlob = await response.blob();

      // Create a File object from the video blob (MOV file)
      const videoFile = new File([videoBlob], 'testVid.mov', {
        type: 'video/quicktime',
      });

      // Create FormData and append the video file
      const formData = new FormData();
      formData.append('file', videoFile);

      setMessage('Uploading file...');

      // Send the file to the backend
      const uploadResponse = await fetch('http://127.0.0.1:5001/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await uploadResponse.json();
      setMessage(!message ? result.message : '');
    } catch (error) {
      console.error('Error uploading file:', error);
      setMessage('Error uploading file');
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
      <Button onClick={handleUpload}>Upload</Button>
      {message && <Typography>{message}</Typography>}
    </Box>
  );
};
