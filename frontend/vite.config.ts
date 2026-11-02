import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the app is served by Vite and every API path goes to the single-origin backend (`uvicorn backend.web:app --port 8000`).
const backend = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "127.0.0.1",
    proxy: {
      "/api": backend,
      "/evidence": backend,
      "/explorer": backend,
      "/twin": backend,
      "/optimization": backend,
      "/health": backend,
    },
  },
});
