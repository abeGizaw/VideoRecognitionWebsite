import { Box, Typography, Card, Button } from '@mui/joy';
import { VideoPreview } from './VideoPreview';
import dynamic from 'next/dynamic';

const ReactMediaRecorder = dynamic(
  () => import('react-media-recorder').then((mod) => mod.ReactMediaRecorder),
  { ssr: false }, // This disables server-side rendering for this component
);

export const Recorder = () => {
  return (
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
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <Typography level='h3' textAlign='center'>
                {status}
              </Typography>
              <Button onClick={startRecording} variant='solid'>
                Start Recording
              </Button>

              <Button onClick={stopRecording} variant='solid' color='danger'>
                Stop Recording
              </Button>
              {status.toLocaleLowerCase() === 'stopped' ? (
                <video src={mediaBlobUrl} controls autoPlay loop />
              ) : (
                <VideoPreview stream={previewStream} blob={mediaBlobUrl} />
              )}
            </Box>
          )}
        />
      </Card>
    </Box>
  );
};
