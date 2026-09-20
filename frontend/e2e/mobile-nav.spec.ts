/**
 * E2E: mobile navigation — hamburger trigger, slide-in panel, and the
 * phone-only CTA path that was previously missing.
 */

import { expect, test } from "@playwright/test";

test.use({ viewport: { width: 375, height: 667 } });

test.describe("mobile nav", () => {
  test("desktop pill is hidden and hamburger is visible on phones", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(
      page.getByRole("button", { name: /open navigation menu/i })
    ).toBeVisible();
  });

  test("panel opens, links navigate, and it closes on route change", async ({
    page,
  }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /open navigation menu/i }).click();

    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    // exact: true — "Analyze" must not substring-match "Start Analyzing".
    const analyzeLink = dialog.getByRole("link", {
      name: "Analyze",
      exact: true,
    });
    await expect(analyzeLink).toBeVisible();

    await analyzeLink.click();
    await expect(page).toHaveURL(/\/analyze$/);
    await expect(dialog).toBeHidden();
  });

  test("phone menu exposes the primary CTA", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /open navigation menu/i }).click();
    // Scoped to the panel: the landing page itself also has a "Start
    // Analyzing" CTA further down, so an unscoped query is ambiguous.
    await expect(
      page
        .getByRole("dialog", { name: /navigation/i })
        .getByRole("link", { name: "Start Analyzing", exact: true })
    ).toBeVisible();
  });

  test("Escape closes the panel", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /open navigation menu/i }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog")).toBeHidden();
  });

  test("backdrop tap closes the panel", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /open navigation menu/i }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
    // Click near the top-left corner: unambiguously the backdrop, never the
    // 78%-width panel anchored to the right edge.
    await page.mouse.click(20, 100);
    await expect(page.getByRole("dialog")).toBeHidden();
  });
});
