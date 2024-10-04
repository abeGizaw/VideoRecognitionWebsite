import { Box } from '@mui/joy';
import type { ReactNode } from 'react';
import { Header } from '../components/Header';
import { JoyLayout } from './JoyLayout';

export const PageTemplate = ({ children }: { children: ReactNode }) => (
  <JoyLayout>
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
      }}
    >
      <Header />
      <Box
        sx={{
          flexGrow: 1,
          marginTop: { xs: '50px', lg: '80px' },
        }}
      >
        {children}
      </Box>
    </Box>
  </JoyLayout>
);
