import { Box, Button, Link } from '@mui/joy';
import { navigationLinks } from '../data/navigationLinks';
import { NavLink } from './NavLink';

interface DesktopNavigationProps {
  path: string;
}

export const DesktopNavigation = (props: DesktopNavigationProps) => {
  const ctaButton = navigationLinks.find((page) => page.cta);

  return (
    <>
      <Box sx={{ display: { xs: 'none', lg: 'contents' } }}>
        {navigationLinks
          .filter((page) => !page.cta)
          .map((page) => (
            <NavLink
              active={page.href === props.path}
              href={page.href}
              key={page.label}
              label={page.label}
            />
          ))}
      </Box>
      <Box sx={{ display: { xs: 'none', lg: 'contents' } }}>
        <Button
          component={Link}
          color='secondary'
          href={ctaButton?.href}
          underline='none'
        >
          {ctaButton?.label}
        </Button>
      </Box>
    </>
  );
};
