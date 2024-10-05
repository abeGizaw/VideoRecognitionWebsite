import devServer from '@hono/vite-dev-server';
import react from '@vitejs/plugin-react';
import vike from 'vike/plugin';
import { defineConfig } from 'vite';

// eslint-disable-next-line no-restricted-syntax
export default defineConfig({
  plugins: [
    vike({ prerender: true }),
    devServer({
      entry: 'hono-entry.ts',

      exclude: [
        /^\/@.+$/,
        /.*\.(ts|tsx|vue)($|\?)/,
        /.*\.(s?css|less)($|\?)/,
        /^\/favicon\.ico$/,
        /.*\.(svg|png)($|\?)/,
        /^\/(public|assets|static)\/.+/,
        /^\/node_modules\/.*/,
      ],

      injectClientScript: false,
    }),
    react({}),
  ],
  assetsInclude: ['**/*.MOV'],
});
