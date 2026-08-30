export function resolveApiBaseUrl(environment) {
  if (environment.VITE_API_BASE_URL) return environment.VITE_API_BASE_URL;

  // Production is served alongside FastAPI, so browser requests stay same-origin.
  return environment.DEV ? "http://localhost:8000" : "";
}

export const API_BASE = resolveApiBaseUrl(import.meta.env);
