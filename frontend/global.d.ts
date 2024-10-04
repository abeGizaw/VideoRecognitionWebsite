import type { PaletteRange } from '@mui/joy/styles';
import type { UserRecord } from 'firebase-admin/auth';

declare module 'hono' {
  interface ContextVariableMap {
    user?: UserRecord | null;
  }
}

declare global {
  namespace Vike {
    interface PageContext {
      user?: UserRecord | null;
    }
  }
}

declare module '@mui/joy/styles' {
  interface ColorPalettePropOverrides {
    secondary: true;
  }

  interface Palette {
    secondary: PaletteRange;
  }
}
