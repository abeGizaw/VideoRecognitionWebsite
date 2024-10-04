
// Temporarily disable Firebase
// See https://firebase.google.com/docs/web/learn-more?hl=fr#config-object
// const firebaseConfig = {
//   apiKey: "",
//   authDomain: "",
//   projectId: "",
//   storageBucket: "",
//   messagingSenderId: "",
//   appId: "",
// };

// let firebaseApp: FirebaseApp | undefined;
// // create a singleton client side firebaseApp
// if (!getApps().length) {
//   firebaseApp = initializeApp(firebaseConfig);
// } else {
//   firebaseApp = getApp();
//   deleteApp(firebaseApp);
//   firebaseApp = initializeApp(firebaseConfig);
// }
//
// const auth = getAuth(firebaseApp);

// As httpOnly cookies are to be used, do not persist any state client side.
// `inMemoryPersistence` is an implementation of Persistence of type 'NONE'.
// auth.setPersistence(inMemoryPersistence);
