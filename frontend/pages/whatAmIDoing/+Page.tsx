import { Box, Button, Card, Typography } from '@mui/joy';
import dynamic from 'next/dynamic';
import { VideoPreview } from '../../components/VideoPreview';

const ReactMediaRecorder = dynamic(
  () => import('react-media-recorder').then((mod) => mod.ReactMediaRecorder),
  { ssr: false }, // This disables server-side rendering for this component
);
export const Page = () => {
  // const handleUpload = (blob: Blob | null) => {
  //   if (blob) {
  //     // Function to send video to the backend for processing
  //     const formData = new FormData();
  //     formData.append('video', blob);
  //     fetch('/upload', {
  //       method: 'POST',
  //       body: formData,
  //     }).then((response) => {
  //       // Handle chatbot conversation based on model output
  //       // Trigger the ChatGPT model to respond
  //     });
  //   }
  // };

  return (
    <Box
      sx={{
        display: 'flex',
        height: '100vh',
        justifyContent: 'space-between',
        gap: '2rem',
        padding: '2rem',
      }}
    >
      {/* Left Side: Video Recording */}
      <Box sx={{ flex: 1 }}>
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
              <Box
                sx={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}
              >
                <Typography level='h3' textAlign='center'>
                  {status}
                </Typography>
                <Button onClick={startRecording} variant='solid'>
                  Start Recording
                </Button>

                <Button onClick={stopRecording} variant='solid' color='danger'>
                  Stop Recording
                </Button>
                <VideoPreview stream={previewStream} blob={mediaBlobUrl} />
              </Box>
            )}
          />
        </Card>
      </Box>

      {/* Right Side: Chatbot Interaction */}
      <Box sx={{ flex: 1 }}>
        <Typography level='h2' textAlign='center' sx={{ marginBottom: '1rem' }}>
          Chat with AI
        </Typography>
        <Card variant='outlined' sx={{ padding: '1rem', height: '100%' }}>
          <div></div>
        </Card>
      </Box>
    </Box>
  );
};
