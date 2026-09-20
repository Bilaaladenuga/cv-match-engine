/**
 * E2E: the analyze flow — the core user journey.
 *
 * Mocks POST /api/matches at the network layer, so the suite exercises the
 * real form, real state handling, and the real report rendering without a
 * backend. Covers: validation gating, loading state, full report rendering,
 * history persistence, and both failure modes (backend error, network down).
 */

import { expect, test } from "@playwright/test";
import { LONG_CV, LONG_JOB, mockMatchEndpoint } from "./fixtures";

const CV_OVER_40 =
  "Jane Okafor\nSenior Backend Engineer with six years of Python experience.";
const JOB_OVER_40 =
  "Senior Platform Engineer\nWe need an engineer with Python and cloud experience.";

test.describe("analyze flow", () => {
  test("submit button is disabled until both texts are long enough", async ({
    page,
  }) => {
    await page.goto("/analyze");
    const submit = page.getByRole("button", { name: /analyze match/i });
    await expect(submit).toBeDisabled();

    await page.fill("#cv", CV_OVER_40);
    await expect(submit).toBeDisabled(); // job still too short

    await page.fill("#job", JOB_OVER_40);
    await expect(submit).toBeEnabled();
  });

  test("shows a loading skeleton, then renders the full report", async ({
    page,
  }) => {
    await mockMatchEndpoint(page, 600); // delay so loading state is visible
    await page.goto("/analyze");

    await page.fill("#cv", LONG_CV);
    await page.fill("#job", LONG_JOB);
    await page.getByRole("button", { name: /analyze match/i }).click();

    // Loading state appears. (Scoped to the button: the page nav also
    // contains the text "Start Analyzing", so a bare text query is ambiguous.)
    await expect(
      page.getByRole("button", { name: /analyzing/i })
    ).toBeVisible();

    // Report renders with distinctive mock content.
    // ("78" renders twice — the score dial and a chart label — both are the
    // score, so assert the first occurrence, the dial.)
    await expect(page.getByText("78").first()).toBeVisible(); // overall score
    await expect(page.getByText(/Good match/i).first()).toBeVisible();
    await expect(page.getByText("E2E-SENTINEL positive factor")).toBeVisible();
    await expect(page.getByText("E2E-SENTINEL negative factor")).toBeVisible();
    await expect(page.getByText("E2E-SENTINEL recommendation")).toBeVisible();
    await expect(page.getByText("E2E-SENTINEL ethics disclaimer")).toBeVisible();

    // Skill evidence rows. Both the mobile cards and the desktop table exist
    // in the DOM (toggled by breakpoint CSS); assert through the table role.
    const skillTable = page.getByRole("table");
    await expect(skillTable.getByText("Machine Learning")).toBeVisible();
    await expect(skillTable.getByText("AWS")).toBeVisible();
  });

  test("analysis is saved to browser history", async ({ page }) => {
    await mockMatchEndpoint(page);
    await page.goto("/analyze");
    await page.fill("#cv", LONG_CV);
    await page.fill("#job", LONG_JOB);
    await page.getByRole("button", { name: /analyze match/i }).click();

    await expect(page.getByText(/saved to browser history/i)).toBeVisible();

    await page.goto("/history");
    await expect(page.getByText(/78/).first()).toBeVisible();
  });

  test("backend error surfaces a readable message", async ({ page }) => {
    await page.route("**/api/matches", (route) =>
      route.fulfill({ status: 500, json: { detail: "Backend exploded" } })
    );
    await page.goto("/analyze");
    await page.fill("#cv", LONG_CV);
    await page.fill("#job", LONG_JOB);
    await page.getByRole("button", { name: /analyze match/i }).click();

    await expect(page.getByText("Backend exploded")).toBeVisible();
    // The skeleton must be gone.
    await expect(page.getByText("E2E-SENTINEL positive factor")).toHaveCount(0);
  });

  test("network failure surfaces the friendly offline message", async ({
    page,
  }) => {
    await page.route("**/api/matches", (route) => route.abort("connectionrefused"));
    await page.goto("/analyze");
    await page.fill("#cv", LONG_CV);
    await page.fill("#job", LONG_JOB);
    await page.getByRole("button", { name: /analyze match/i }).click();

    await expect(
      page.getByText(/cannot reach the analysis service/i)
    ).toBeVisible();
  });
});
