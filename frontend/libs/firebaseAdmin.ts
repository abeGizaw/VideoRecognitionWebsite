import 'dotenv/config';
import {
  type App,
  applicationDefault,
  getApp,
  getApps,
  initializeApp,
} from 'firebase-admin/app';
import { getAuth } from 'firebase-admin/auth';

let firebaseAdmin: App | undefined;

if (!getApps().length) {
  firebaseAdmin = initializeApp({
    credential: applicationDefault(),
  });
} else {
  firebaseAdmin = getApp();
}

export { firebaseAdmin, getAuth };
