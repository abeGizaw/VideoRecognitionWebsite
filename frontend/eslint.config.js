import globals from 'globals';
import pluginJs from '@eslint/js';
import tseslint from 'typescript-eslint';
import pluginReact from 'eslint-plugin-react';
import prettier from 'eslint-config-prettier';

export default [
  { files: ['**/*.{js,mjs,cjs,ts,jsx,tsx}'] },
  { languageOptions: { globals: globals.browser } },
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,
  pluginReact.configs.flat.recommended,
  prettier,
  {
    rules: {
      'react/react-in-jsx-scope': 'off',
      'no-restricted-syntax': [
        'error',
        {
          selector: 'FunctionExpression',
          message: 'Prefer arrow functions',
        },
        {
          selector: 'FunctionDeclaration',
          message: 'Prefer arrow functions',
        },
      ],
    },
  },
];
