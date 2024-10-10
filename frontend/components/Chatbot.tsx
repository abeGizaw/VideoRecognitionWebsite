import { Box, Typography, Card } from '@mui/joy';

export const Chatbot = () => {
  return (
    <Box sx={{ flex: 1 }}>
      <Typography level='h2' textAlign='center' sx={{ marginBottom: '1rem' }}>
        Chat with AI
      </Typography>
      <Card variant='outlined' sx={{ padding: '1rem', height: '100%' }}>
        <div>hello world</div>
      </Card>
    </Box>
  );
};
