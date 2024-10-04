import { ElementType } from 'react';
import { Box } from '@mui/joy';

interface IconProps {
  component: ElementType;
}

export const Icon = (props: IconProps) => (
  <Box className='ui-icon' sx={{ display: 'flex' }}>
    <props.component />
  </Box>
);
