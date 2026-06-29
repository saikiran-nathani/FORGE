/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute URL of the FORGE backend API (no trailing slash). Empty in dev. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
