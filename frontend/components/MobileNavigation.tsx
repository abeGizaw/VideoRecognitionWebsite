import {
  Box,
  Button,
  DialogContent,
  DialogTitle,
  Drawer,
  IconButton,
  Link,
  ModalClose,
} from '@mui/joy';
import { useState } from 'react';
import { navigationLinks } from '../data/navigationLinks';
import { NavLink } from './NavLink';
import { MdMenu } from 'react-icons/md';
import { Icon } from './Icon';

interface MobileNavigationProps {
  path: string;
}

export const MobileNavigation = (props: MobileNavigationProps) => {
  const [open, setOpen] = useState(false);
  const ctaButton = navigationLinks.find((page) => page.cta);

  return (
    <>
      <Box sx={{ display: { lg: 'none' } }}>
        <IconButton onClick={() => setOpen(true)} variant='plain'>
          <Icon component={MdMenu} />
        </IconButton>
      </Box>
      <Drawer
        anchor='right'
        open={open}
        onClose={() => setOpen(false)}
        slotProps={{
          content: {
            sx: (theme) => ({
              width: 'max-content',
              background: theme.palette.background.level1,
            }),
          },
        }}
      >
        <ModalClose />
        <DialogTitle>Menu</DialogTitle>
        <DialogContent sx={{ marginInline: 2 }}>
          {navigationLinks
            .filter((page) => !page.cta)
            .map((page) => (
              <NavLink
                active={page.href === props.path}
                href={page.href}
                key={page.label}
                label={page.label}
                sx={{ marginBlock: 1 }}
              />
            ))}
          <Button
            component={Link}
            color='secondary'
            href={ctaButton?.href}
            underline='none'
            sx={{ marginTop: 2 }}
          >
            {ctaButton?.label}
          </Button>
        </DialogContent>
      </Drawer>
    </>
  );
};
