// src/config/time.ts
export const MAX_TIME_SECONDS = 600;

// Presets shown in the UI
export const TIME_PRESETS = [
  { label: "0–1s", start: 0, end: 1 },
  { label: "1–10s", start: 1, end: 10 },
  { label: "10–60s", start: 10, end: 60 },
  { label: "60–300s", start: 60, end: 300 },
  { label: `Full (0–${MAX_TIME_SECONDS}s)`, start: 0, end: MAX_TIME_SECONDS },
] as const;
