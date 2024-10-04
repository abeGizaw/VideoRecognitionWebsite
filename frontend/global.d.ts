import type { UserRecord } from "firebase-admin/auth";

declare global {
  namespace Vike {
    interface PageContext {
      user?: UserRecord | null;
    }
  }
}

declare module "express" {
  interface Request {
    user?: UserRecord | null;
  }
}

export {};
