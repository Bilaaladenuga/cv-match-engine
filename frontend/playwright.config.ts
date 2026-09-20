import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright e2e configuration.
 *
 * Tests mock the backend with page.route(), so no real API is needed. The
 * suite runs against a PRODUCTION build (`next build` output served by
 * `next start`): dev-mode on-demand compilation is slow enough on Windows
 * to eat the per-test timeout, and production is what actually ships.
 * If no build exists, run `npm run build` first.
 *
 * Chromium only: the assertions are behavioural, not browser-specific, and
 * one browser keeps local runs fast.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  fullyParallel: false,
  retries: 0,
  workers: 1,
  use: {
    baseURL: "http://localhost:3000",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run start -- --port 3000",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
