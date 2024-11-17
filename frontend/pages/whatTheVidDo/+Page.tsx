import { Box, Button, Typography } from '@mui/joy';
import { useCallback, useEffect, useState } from 'react';
// React Dropzone uses a CommonJS export while Vite expects ES Module exports, so we need to import it like this
import * as pkg from 'react-dropzone';
import { DragNDrop } from '../../components/DragNDrop';
import { TextToSpeech } from '../../components/TexToSpeech';
import { VideoInfo } from '../../components/VideoInfo';
const { useDropzone } = pkg;

export const Page = () => {
  const [message, setMessage] = useState<string>('');
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string>('');
  const [textToSpeechBox, setTextToSpeechBox] = useState<boolean>(false);

  // Callback to handle the file drop event in our 'Dropzone' component
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file.type.startsWith('video/')) {
        setVideoFile(file);
        setMessage(`Selected video: ${file.name}`);
        const previewUrl = URL.createObjectURL(file);
        setVideoPreview(previewUrl);
        setTextToSpeechBox(false);
      } else {
        setMessage('Please select a valid video file.');
      }
    }
  }, []);

  useEffect(() => {
    if (videoPreview) {
      return () => {
        URL.revokeObjectURL(videoPreview);
      };
    }
  }, [videoPreview]);


  const handleUpload = async () => {
    if (videoFile) {
      try {
        // Create a FormData object to send the file to the server
        const formData = new FormData();
        formData.append('file', videoFile);
        setMessage('Uploading file...');

        // const uploadResponse = await fetch(
        //   `${"https://my-backend-app-1001376648512.us-central1.run.app"}/upload`,
        //   {
        //     method: 'POST',
        //     body: formData,
        //     headers: {
        //       Accept: 'application/json',
        //     },
        //   },
        // );

        const uploadResponse = await fetch(
          `http://127.0.0.1:8080/upload`,  // Flask server local URL
          {
            method: 'POST',
            body: formData,
            headers: {
              Accept: 'application/json',
            },
          },
        );

        if (uploadResponse.ok) {
          const result = await uploadResponse.json();
          setMessage(result.message);
          setTextToSpeechBox(true);
        } else {
          setMessage('Failed to upload file.');
        }
      } catch (error) {
        console.error('Error uploading file:', error);
        setMessage('Failed to upload file.');
      }
    }
  };

  // React Dropzone hook to get the props for the dropzone area
  const { getRootProps, isDragActive, getInputProps } = useDropzone({
    onDrop,
    accept: 'video/*',
    multiple: false,
  });

  const handleRemove = () => {
    setVideoFile(null);
    setMessage('');
  };

  return (
    <Box
      sx={{
        width: 'fit-content',
        margin: 'auto',
        minHeight: '100vh',
        alignItems: 'center',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        paddingTop: '2rem',
      }}
    >
      <Typography level='h1'>Upload Video</Typography>

      {/* Drag and Drop Area */}
      <DragNDrop
        rootProps={getRootProps}
        inputProps={getInputProps}
        isDragActive={isDragActive}
      />

      {/* Video preview and remove button */}
      {videoFile && (
        <Box sx={{ textAlign: 'center' }}>
          <Typography>Selected video: {videoFile.name}</Typography>
          <video width='300' controls key={videoPreview}>
            <source src={videoPreview} type={videoFile.type} />
            Your browser does not support the video tag.
          </video>
          <Button onClick={handleRemove} sx={{ marginTop: '1rem' }}>
            Remove Video
          </Button>
        </Box>
      )}

      {/* Upload button */}
      <Button onClick={handleUpload} disabled={!videoFile}>
        Upload
      </Button>
      <VideoInfo message={message} textToSpeechBox={textToSpeechBox} />
    </Box>
  );
};
