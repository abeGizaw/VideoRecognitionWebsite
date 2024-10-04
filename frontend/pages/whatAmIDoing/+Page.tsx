import { Box, Button, Typography } from '@mui/joy';

export const Page = () => {
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
      <Typography level='h1'>Record Me</Typography>
      <Button>Record</Button>
    </Box>
  );
};
