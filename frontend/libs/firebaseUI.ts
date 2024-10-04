import type { UserCredential } from 'firebase/auth';
import firebase from 'firebase/compat/app';
import type * as firebaseui from 'firebaseui';

type SessionLogin = (authResult: UserCredential) => Promise<void>;

const startFirebaseUI = (
  ui: firebaseui.auth.AuthUI,
  sessionLogin?: SessionLogin,
) => {
  // Please read firebaseui docs at https://github.com/firebase/firebaseui-web
  const uiConfig = {
    callbacks: {
      signInSuccessWithAuthResult(authResult: UserCredential) {
        if (sessionLogin) {
          void sessionLogin(authResult);
          // Don't redirect after firebase client successfully sign-in, let vike handle the rest.
          return false;
        }
        return true;
      },
    },
    signInFlow: 'popup',
    signInOptions: [
      // Disable "Email enumeration protection" to be able to login with registered email address
      // https://console.firebase.google.com/u/1/project/{project-id}/authentication/settings
      firebase.auth.EmailAuthProvider.PROVIDER_ID,
      firebase.auth.GoogleAuthProvider.PROVIDER_ID,
    ],
  };
  return ui.start('#firebaseui-auth-container', uiConfig);
};

export { startFirebaseUI };
