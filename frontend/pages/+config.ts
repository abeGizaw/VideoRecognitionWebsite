import vikeReact from 'vike-react/config';
import type { Config } from 'vike/types';
import { PageTemplate as Layout } from '../layouts/PageTemplate';

// Default config (can be overridden by pages)
// https://vike.dev/config

export default {
  // https://vike.dev/Layout
  Layout,

  // https://vike.dev/head-tags
  title: 'WhatTheyDoing',
  description: '',

  passToClient: ['user'],
  extends: vikeReact,
} satisfies Config;
