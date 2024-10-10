import { Box, Typography } from '@mui/joy';
import { useDropzone } from 'react-dropzone';

interface DragNDropProps {
  rootProps: ReturnType<typeof useDropzone>['getRootProps'];
  inputProps: ReturnType<typeof useDropzone>['getInputProps'];
  isDragActive: boolean;
}
export const DragNDrop = ({
  rootProps,
  inputProps,
  isDragActive,
}: DragNDropProps) => {
  return (
    <Box
      {...rootProps()}
      sx={{
        border: '2px dashed gray',
        padding: '2rem',
        textAlign: 'center',
        width: '300px',
        borderRadius: '10px',
        backgroundColor: '#f9f9f9',
        cursor: 'pointer',
      }}
    >
      <input
        {...(inputProps() as React.InputHTMLAttributes<HTMLInputElement>)}
      />{' '}
      {isDragActive ? (
        <Typography>Drop the video here...</Typography>
      ) : (
        <Typography>
          Drag and drop a video here, or click to select one
        </Typography>
      )}
    </Box>
  );
};
