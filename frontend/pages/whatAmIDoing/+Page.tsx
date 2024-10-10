import { Box } from '@mui/joy';
import { Recorder } from '../../components/Recorder';
import { Chatbot } from '../../components/Chatbot';
import { useState } from 'react';
export interface Message {
  text: string;
  isUser: boolean;
}

export const Page = () => {
  const [messages, setMessages] = useState<Message[]>([
    { text: 'Hello! How can I help you?', isUser: false },
  ]);
  
  return (
    <Box
      sx={{
        display: 'flex',
        height: '100vh',
        justifyContent: 'space-between',
        gap: '2rem',
        padding: '2rem',
        flexDirection: { xs: 'column', lg: 'row' },
      }}
    >
      <Recorder setMessage={setMessages} />
      <Chatbot messages={messages} />
    </Box>
  );
};
