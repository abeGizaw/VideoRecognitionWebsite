import { Link } from '@mui/joy';
import { NavigationLink } from '../data/navigationLinks';
import { SxProps } from '@mui/joy/styles/types';

interface NavLinkProps extends Omit<NavigationLink, 'cta'> {
  active: boolean;
  sx?: SxProps;
}

export const NavLink = (props: NavLinkProps) => (
  <Link href={props.href} color='neutral' sx={props.sx}>
    {props.active ? <b>{props.label}</b> : props.label}
  </Link>
);
