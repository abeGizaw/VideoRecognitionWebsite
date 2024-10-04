<!--bati:start section="document"-->

<!--bati:start section="intro"-->

Generated with [Bati](https://batijs.dev) ([version 292](https://www.npmjs.com/package/create-bati/v/0.0.292)) using this command:

```sh
--npm create bati -- --react --firebase-auth --express --prettier
```

<!--bati:start section="TOC"-->

## Contents

* [React](#react)

  * [`/pages/+config.ts`](#pagesconfigts)
  * [Routing](#routing)
  * [`/pages/_error/+Page.jsx`](#pages_errorpagejsx)
  * [`/pages/+onPageTransitionStart.ts` and `/pages/+onPageTransitionEnd.ts`](#pagesonpagetransitionstartts-and-pagesonpagetransitionendts)
  * [SSR](#ssr)
  * [HTML Streaming](#html-streaming)

* [*Firebase*](#firebase)

<!--bati:end section="TOC"-->

<!--bati:end section="intro"-->

<!--bati:start section="features"-->

<!--bati:start category="UI Framework" flag="react"-->

## React

This app is ready to start. It's powered by [Vike](https://vike.dev) and [React](https://react.dev/learn).

### `/pages/+config.ts`

Such `+` files are [the interface](https://vike.dev/config) between Vike and your code. It defines:

* A default [`<Layout>` component](https://vike.dev/Layout) (that wraps your [`<Page>` components](https://vike.dev/Page)).
* A default [`title`](https://vike.dev/title).
* Global [`<head>` tags](https://vike.dev/head-tags).

### Routing

[Vike's built-in router](https://vike.dev/routing) lets you choose between:

* [Filesystem Routing](https://vike.dev/filesystem-routing) (the URL of a page is determined based on where its `+Page.jsx` file is located on the filesystem)
* [Route Strings](https://vike.dev/route-string)
* [Route Functions](https://vike.dev/route-function)

### `/pages/_error/+Page.jsx`

The [error page](https://vike.dev/error-page) which is rendered when errors occur.

### `/pages/+onPageTransitionStart.ts` and `/pages/+onPageTransitionEnd.ts`

The [`onPageTransitionStart()` hook](https://vike.dev/onPageTransitionStart), together with [`onPageTransitionEnd()`](https://vike.dev/onPageTransitionEnd), enables you to implement page transition animations.

### SSR

SSR is enabled by default. You can [disable it](https://vike.dev/ssr) for all your pages or only for some pages.

### HTML Streaming

You can enable/disable [HTML streaming](https://vike.dev/stream) for all your pages, or only for some pages while still using it for others.

<!--bati:end category="UI Framework" flag="react"-->

<!--bati:start category="Auth" flag="firebase-auth"-->

## *Firebase*

* You first need to **[Create a Firebase project](https://firebase.google.com/docs/web/setup#create-project)**.
* Then register your app in the firebase console. **[Register your app](https://firebase.google.com/docs/web/setup#register-app)**
* Copy Your web app's Firebase configuration and paste in `/pages/+client.ts` Example :

```ts
const firebaseConfig = {
  apiKey: "",
  authDomain: "",
  projectId: "",
  storageBucket: "",
  messagingSenderId: "",
  appId: "",
};
```

* Download Your Firebase service account from [Your Firebase Project Settings > Service accounts](https://console.firebase.google.com/u/0/project/{firebase-project-id}/settings/serviceaccounts/adminsdk)
* Rename to service-account.json and move it to folder `/firebase/`.
* Read more about Firebase Auth at official [firebase auth docs](https://firebase.google.com/docs/auth)
* Read FirebaseUI at [firebaseui-web docs](https://github.com/firebase/firebaseui-web?tab=readme-ov-file#using-firebaseui-for-authentication)

<!--bati:end category="Auth" flag="firebase-auth"-->

<!--bati:end section="features"-->

<!--bati:end section="document"-->
