import { Box, Button, Typography } from '@mui/joy';
import { useCallback, useEffect, useState } from 'react';
// React Dropzone uses a CommonJS export while Vite expects ES Module exports, so we need to import it like this
import * as pkg from 'react-dropzone';
import { DragNDrop } from '../../components/DragNDrop';
import { VideoInfo } from '../../components/VideoInfo';
import { MessageTypes } from '../../data/constants';
const { useDropzone } = pkg;

export type Message = {
  message: string;
  type: string;
}

export const Page = () => {
  const [message, setMessage] = useState<Message>({ message: '', type: '' });
  const [classificationMessage, setClassificationMessage] = useState<Message>({ message: '', type: '' });
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string>('');
  const [textToSpeechBox, setTextToSpeechBox] = useState<boolean>(false);

  // Callback to handle the file drop event in our 'Dropzone' component
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file.type.startsWith('video/')) {
        setVideoFile(file);
        setMessage({ message: `Selected video: ${file.name}`, type: MessageTypes.PRE });
        const previewUrl = URL.createObjectURL(file);
        setVideoPreview(previewUrl);
        setTextToSpeechBox(false);
      } else {
        setMessage({ message: `Please select a valid video file.`, type: MessageTypes.ERROR });
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
        setMessage({ message: 'Uploading file...', type: MessageTypes.PRE });

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
          setClassificationMessage({ message: result.message, type: MessageTypes.POST });
          setMessage({ message: '', type: MessageTypes.PRE });
          setTextToSpeechBox(true);
        } else {
          setMessage({ message: 'Failed to upload file.', type: MessageTypes.ERROR });
        }
      } catch (error) {
        console.error('Error uploading file:', error);
        setMessage({ message: 'Failed to upload file.', type: MessageTypes.ERROR });
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
    setMessage({ message: '', type: MessageTypes.PRE });
    setClassificationMessage({ message: '', type: MessageTypes.PRE });
    setTextToSpeechBox(false);
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
      <VideoInfo displayMessage={message} resultMessage={classificationMessage} textToSpeechBox={textToSpeechBox} />
    </Box>
  );
};
