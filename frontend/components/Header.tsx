import { Box, Link } from '@mui/joy';
import { DesktopNavigation } from './DesktopNavigation';
import { usePageContext } from 'vike-react/usePageContext';
import { MobileNavigation } from './MobileNavigation';
import { constants } from '../data/constants';
import logo from '../assets/logo.png';

export const Header = () => {
  const pageContext = usePageContext();

  return (
    <Box
      sx={(theme) => ({
        position: 'fixed',
        width: '100%',
        zIndex: constants.HEADER_Z_INDEX,
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingInline: 5,
        background: theme.palette.background.level1,
        boxShadow: theme.shadow.xs,
      })}
    >
      <Link href='/'>
        <Box
          component='img'
          src={logo}
          alt='What it doooo'
          sx={{ height: { xs: '50px', lg: '80px' } }}
        />
      </Link>
      <DesktopNavigation path={pageContext.urlPathname} />
      <MobileNavigation path={pageContext.urlPathname} />
    </Box>
  );
};
