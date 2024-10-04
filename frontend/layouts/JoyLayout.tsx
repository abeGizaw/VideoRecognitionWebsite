import {
  CssBaseline,
  CssVarsProvider,
  extendTheme,
  GlobalStyles,
  Theme,
} from '@mui/joy';
import { type ReactNode, useEffect, useState } from 'react';
import { Interpolation } from '@emotion/react';

const primary = {
  50: '#e8f8ec', // Lightest green
  100: '#d1f1d9',
  200: '#a4e4b2', // Lighter green
  300: '#70d788', // Main green shade
  400: '#4cc85e', // Slightly darker green
  500: '#2db940', // Primary green
  600: '#24912e',
  700: '#1a6a21', // Dark green
  800: '#114216', // Darker green
  900: '#0c3011', // Almost black green
  solidBg: 'var(--joy-palette-primary-400)',
  solidActiveBg: 'var(--joy-palette-primary-500)',
  outlinedBorder: 'var(--joy-palette-primary-500)',
  outlinedColor: 'var(--joy-palette-primary-700)',
  outlinedActiveBg: 'var(--joy-palette-primary-100)',
  softColor: 'var(--joy-palette-primary-800)',
  softBg: 'var(--joy-palette-primary-200)',
  softActiveBg: 'var(--joy-palette-primary-300)',
  plainColor: 'var(--joy-palette-primary-700)',
  plainActiveBg: 'var(--joy-palette-primary-100)',
};


const secondary = {
  50: '#f6e8ff', // Lightest purple
  100: '#e6ccff',
  200: '#c299ff', // Lighter purple
  300: '#9966ff', // Main purple shade
  400: '#8041ff', // Slightly darker purple
  500: '#6623e0', // Primary purple
  600: '#5017b3',
  700: '#3a0d85', // Dark purple
  800: '#28085c', // Darker purple
  900: '#19063a', // Almost black purple
  solidBg: 'var(--joy-palette-secondary-400)',
  solidHoverBg: 'var(--joy-palette-secondary-600)',
  solidActiveBg: 'var(--joy-palette-secondary-500)',
  outlinedBorder: 'var(--joy-palette-secondary-500)',
  outlinedColor: 'var(--joy-palette-secondary-700)',
  outlinedActiveBg: 'var(--joy-palette-secondary-100)',
  softColor: 'var(--joy-palette-secondary-800)',
  softBg: 'var(--joy-palette-secondary-200)',
  softActiveBg: 'var(--joy-palette-secondary-300)',
  plainColor: 'var(--joy-palette-secondary-700)',
  plainActiveBg: 'var(--joy-palette-secondary-100)',
  solidColor: 'white',
};


const theme = extendTheme({
  colorSchemes: {
    light: {
      palette: {
        primary,
        secondary,
      },
    },
    dark: {
      palette: {
        primary,
        secondary,
      },
    },
  },
  fontFamily: {
    display: 'DM Sans',
    body: 'DM Sans',
  },
});

const iconConfig: Interpolation<Theme> = {
  '.ui-icon svg': {
    color: 'var(--Icon-color)',
    margin: 'var(--Icon-margin)',
    fontSize: 'var(--Icon-fontSize, 20px)',
    width: '1em',
    height: '1em',
  },
};

export const JoyLayout = ({ children }: { children: ReactNode }) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  return (
    <CssVarsProvider
      theme={theme}
      defaultMode='system'
      disableNestedContext={true}
    >
      <CssBaseline />
      <GlobalStyles styles={iconConfig} />
      {mounted ? children : <></>}
    </CssVarsProvider>
  );
};
