export interface NavigationLink {
  href: string;
  label: string;
  cta?: boolean;
}

export const navigationLinks: NavigationLink[] = [
  { href: '/', label: 'Home' },
  { href: '/whatAmIDoing', label: 'Record Me' },
  { href: '/whatTheVidDo', label: 'Classify Video', cta: true },
];
