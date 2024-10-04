import 'firebaseui/dist/firebaseui.css';
import { getAuth, type UserCredential } from 'firebase/auth';
import * as firebaseui from 'firebaseui';
import { useEffect, useState } from 'react';
import { reload } from 'vike/client/router';
import { startFirebaseUI } from '../../libs/firebaseUI';

export const Page = () => {
  const [error, setError] = useState('');

  const sessionLogin = async (authResult: UserCredential) => {
    const idToken = (await authResult.user.getIdToken()) || '';
    try {
      const response = await fetch('/api/sessionLogin', {
        method: 'POST',
        body: JSON.stringify({ idToken }),
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (response.ok) {
        await reload();
      } else {
        setError(response.statusText);
      }
      await getAuth().signOut();
    } catch (err) {
      console.log('error :', err);
    }
  };

  useEffect(() => {
    const ui =
      firebaseui.auth.AuthUI.getInstance() ||
      new firebaseui.auth.AuthUI(getAuth());
    if (!error) {
      startFirebaseUI(ui, sessionLogin);
    }
  }, [error]);

  return (
    <>
      <div id='firebaseui-auth-container' />
      {error && (
        <>
          <div style={{ color: 'red' }}>There was an error : {error}</div>
          <button onClick={() => setError('')} type='button'>
            Try Again
          </button>
        </>
      )}
    </>
  );
};
