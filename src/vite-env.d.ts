/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_GIT_COMMIT?: string;
  readonly VITE_REPOSITORY_URL?: string;
  readonly VITE_BENCHMARK_URL?: string;
}
