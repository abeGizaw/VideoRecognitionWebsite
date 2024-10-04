import { vikeHandler } from './server/vike-handler';
import { Hono } from 'hono';
import { createHandler } from '@universal-middleware/hono';

const app = new Hono();

// Not using Firebase at the moment
// app.use(createMiddleware(firebaseAuthMiddleware)());
// app.post("/api/sessionLogin", createHandler(firebaseAuthLoginHandler)());
// app.post("/api/sessionLogout", createHandler(firebaseAuthLogoutHandler)());

/**
 * Vike route
 *
 * @link {@see https://vike.dev}
 **/
app.all('*', createHandler(vikeHandler)());

// eslint-disable-next-line no-restricted-syntax
export default app;
