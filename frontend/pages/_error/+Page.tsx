import { Typography } from '@mui/joy';
import { usePageContext } from 'vike-react/usePageContext';

export const Page = () => {
  const { is404 } = usePageContext();
  if (is404) {
    return (
      <>
        <Typography level='h1'>404 Page not found</Typography>
        <Typography>This page could not be found</Typography>
      </>
    );
  }
  return (
    <>
      <Typography level='h1'>500 Internal Server Error</Typography>
      <Typography>Something went wrong.</Typography>
    </>
  );
};
