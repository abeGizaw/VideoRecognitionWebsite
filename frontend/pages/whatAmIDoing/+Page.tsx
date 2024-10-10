import { Box } from '@mui/joy';
import { Recorder } from '../../components/Recorder';
import { Chatbot } from '../../components/Chatbot';

export const Page = () => {
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
      <Recorder />
      <Chatbot />
    </Box>
  );
};
