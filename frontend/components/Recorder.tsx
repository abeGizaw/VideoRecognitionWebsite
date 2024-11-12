import { Box, Typography, Card, Button } from '@mui/joy';
import { VideoPreview } from './VideoPreview';
import dynamic from 'next/dynamic';
import { useState } from 'react';
import { Message } from '../pages/whatAmIDoing/+Page';

const ReactMediaRecorder = dynamic(
  () => import('react-media-recorder').then((mod) => mod.ReactMediaRecorder),
  { ssr: false }, // This disables server-side rendering for this component
);

export interface RecorderProps {
  setMessage: (response: (prevItems: Message[]) => Message[]) => void;
}
export const Recorder = ({ setMessage }: RecorderProps) => {
  const [videoPresent, setVideoPresent] = useState<boolean>(false);
  const [blobUrl, setBlobUrl] = useState<string | undefined>();

  const handleStopRecording = (stopRecording: () => void) => {
    stopRecording();
    setVideoPresent(true);
  };

  const handleUpload = async () => {
    if (blobUrl) {
      try {
        // Fetch the Blob data from the blobUrl
        const blob = await fetch(blobUrl).then((res) => res.blob());

        // Create a FormData object to send the blob to the server
        const formData = new FormData();
        formData.append('file', blob, 'video.mp4');

        setMessage((prevItems: Message[]) => [
          ...prevItems,
          { text: 'Uploading...', isUser: true },
          { text: 'Let me process that...', isUser: false },
        ]);


        //`${'https://my-backend-app-1001376648512.us-central1.run.app'}/upload?source=chatBot`,
     
        // const uploadResponse = await fetch(
        //   `${import.meta.env.VITE_BACKEND_URL}/upload?source=chatBot`,
        //   {
        //     method: 'POST',
        //     body: formData,
        //     headers: {
        //       Accept: 'application/json',
        //     },
        //   },
        // );
        const uploadResponse = await fetch(
          `http://127.0.0.1:8080/upload?source=chatBot`,
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
          setMessage((prevItems: Message[]) => {
            const newItems = [...prevItems];
            newItems[newItems.length - 2] = {
              text: result.message,
              isUser: true,
            };
            newItems[newItems.length - 1] = {
              text: result.message,
              isUser: false,
            };
            return newItems;
          });
        } else {
          setMessage((prevItems: Message[]) => [
            ...prevItems,
            { text: 'Error uploading video', isUser: false },
          ]);
        }
      } catch (error) {
        console.error('Error uploading video:', error);
        setMessage((prevItems: Message[]) => [
          ...prevItems,
          { text: 'Error uploading video', isUser: false },
        ]);
      }
    } else {
      setMessage((prevItems: Message[]) => [
        ...prevItems,
        { text: 'No video to upload', isUser: false },
      ]);
    }
  };

  return (
    <Box sx={{ flex: 1, alignItems: 'center' }}>
      <Typography level='h2' textAlign='center' sx={{ marginBottom: '1rem' }}>
        Video Recording
      </Typography>
      <Card variant='outlined' sx={{ padding: '1rem' }}>
        <ReactMediaRecorder
          video
          audio={false}
          render={({
            startRecording,
            mediaBlobUrl,
            status,
            stopRecording,
            previewStream,
          }) => (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <Typography level='h3' textAlign='center'>
                {status}
              </Typography>
              <Button onClick={startRecording} variant='solid'>
                Start Recording
              </Button>

              <Button
                onClick={() => handleStopRecording(stopRecording)}
                variant='solid'
                color='danger'
              >
                Stop Recording
              </Button>
              {status.toLocaleLowerCase() === 'stopped' ? (
                <video
                  onLoadedMetadata={() => setBlobUrl(mediaBlobUrl)}
                  src={mediaBlobUrl}
                  controls
                  autoPlay
                  loop
                />
              ) : (
                <VideoPreview stream={previewStream} blob={mediaBlobUrl} />
              )}
            </Box>
          )}
        />
      </Card>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          marginTop: '2rem',
        }}
      >
        <Button
          variant='solid'
          onClick={handleUpload}
          color='primary'
          disabled={!videoPresent}
        >
          Upload
        </Button>
      </Box>
    </Box>
  );
};
