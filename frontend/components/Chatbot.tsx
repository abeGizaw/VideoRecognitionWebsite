import { Box, Typography, Card } from '@mui/joy';
import { Message } from '../pages/whatAmIDoing/+Page';

interface ChatbotProps {
  messages: Message[];
}
export const Chatbot = ({ messages }: ChatbotProps) => {
  return (
    <Box sx={{ flex: 1 }}>
      <Typography level='h2' textAlign='center' sx={{ marginBottom: '1rem' }}>
        Chat with AI
      </Typography>
      <Card variant='outlined' sx={{ padding: '1rem', height: '100%' }}>
        {messages.map((message, index) => (
          <Card
            key={index}
            sx={{
              alignSelf: message.isUser ? 'flex-end' : 'flex-start',
              backgroundColor: message.isUser ? '#e0ffe0' : '#e0e0ff',
              padding: '0.5rem 1rem',
              borderRadius: '15px',
              maxWidth: '70%',
            }}
          >
            {message.text.split('\n').map((line, i) => (
              <div key={i}>{line}</div>
            ))}
          </Card>
        ))}
      </Card>
    </Box>
  );
};
