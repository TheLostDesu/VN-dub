import { fileURLToPath, URL } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The page is served by FastAPI (and shown in a pywebview window), so every asset path is relative.
// The i18n tables live in the repository (app/i18n), outside this folder: allow Vite to read them.
const repoRoot = fileURLToPath(new URL("../../", import.meta.url));
const i18nDir = fileURLToPath(new URL("../../app/i18n", import.meta.url));

export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: {
    alias: { "@i18n": i18nDir },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    fs: { allow: [repoRoot] },
    proxy: {
      "/api": { target: "http://127.0.0.1:8765", changeOrigin: false, ws: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
});
