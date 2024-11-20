import {
  Box,
  Option,
  Select,
  Typography,
  useColorScheme,
} from '@mui/joy';

export const Page = () => {
  const { mode, setMode } = useColorScheme();

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
      <Typography level='h1'>What Tha Vid Do 🎥</Typography>
      <Select value={mode} onChange={(_, newMode) => setMode(newMode)}>
        <Option value='system'>System</Option>
        <Option value='light'>Light</Option>
        <Option value='dark'>Dark</Option>
      </Select>
    </Box>
  );
};
