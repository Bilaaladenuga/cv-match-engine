/**
 * Shared e2e helpers.
 *
 * The mocked MatchReport mirrors backend/app/schemas/match.py — keep field
 * names in sync when the schema changes. Text strings are deliberately
 * distinctive so tests can assert on them without depending on CSS markup.
 */

import type { Page } from "@playwright/test";
import type { MatchReport } from "../lib/types";

export const MOCK_REPORT: MatchReport = {
  match_id: null,
  model_version: "match-model-v0.5.0-e2e",
  overall_score: 0.78,
  overall_percent: 78,
  band: "Good match",
  components: [
    {
      name: "skills",
      raw_score: 0.85,
      weight: 0.4,
      weighted: 0.34,
      evidence: "E2E-SENTINEL skills evidence",
    },
    {
      name: "semantic",
      raw_score: 0.81,
      weight: 0.25,
      weighted: 0.2,
      evidence: "E2E-SENTINEL semantic evidence",
    },
    {
      name: "experience",
      raw_score: 0.7,
      weight: 0.2,
      weighted: 0.14,
      evidence: "E2E-SENTINEL experience evidence",
    },
    {
      name: "education",
      raw_score: 1.0,
      weight: 0.1,
      weighted: 0.1,
      evidence: "E2E-SENTINEL education evidence",
    },
    {
      name: "certifications",
      raw_score: 0.0,
      weight: 0.05,
      weighted: 0.0,
      evidence: "E2E-SENTINEL certifications evidence",
    },
  ],
  weights: {
    skills: 0.4,
    semantic: 0.25,
    experience: 0.2,
    education: 0.1,
    certifications: 0.05,
  },
  positive_factors: ["E2E-SENTINEL positive factor"],
  negative_factors: ["E2E-SENTINEL negative factor"],
  recommendations: ["E2E-SENTINEL recommendation"],
  disclaimer: "E2E-SENTINEL ethics disclaimer",
  skill_evidence: [
    {
      skill: "Python",
      status: "matched",
      candidate_skill: "Python",
      strength: "strong",
      in_skills_section: true,
      in_work_history: true,
      estimated_months: 48,
      sources: ["skills section", "work history"],
    },
    {
      skill: "Machine Learning",
      status: "partial",
      candidate_skill: "Machine Learning",
      strength: "moderate",
      in_skills_section: true,
      in_work_history: false,
      estimated_months: null,
      sources: ["skills section"],
    },
    {
      skill: "AWS",
      status: "missing",
      candidate_skill: null,
      strength: "absent",
      in_skills_section: false,
      in_work_history: false,
      estimated_months: null,
      sources: [],
    },
  ],
  ml_details: {
    label: "Good Fit",
    fit_score: 0.81,
    probabilities: { "Good Fit": 0.81, "Potential Fit": 0.14, "No Fit": 0.05 },
    ml_version: "match-model-v0.5.0",
    confidence: 0.81,
    confidence_note: "High",
    explanation: {
      method: "top-contribution",
      factors: [
        {
          feature: "skill_overlap_ratio",
          label: "E2E-SENTINEL factor label",
          contribution: 0.32,
          direction: "helps",
          value: 0.85,
          reference: 0.55,
          detail: "E2E-SENTINEL factor detail",
        },
      ],
      cautions: ["E2E-SENTINEL caution"],
      disclaimer: "E2E-SENTINEL ml disclaimer",
    },
  },
};

/** Intercept the match endpoint, returning the mock report after `delayMs`. */
export async function mockMatchEndpoint(page: Page, delayMs = 0): Promise<void> {
  await page.route("**/api/matches", async (route) => {
    if (delayMs > 0) await page.waitForTimeout(delayMs);
    await route.fulfill({ status: 201, json: MOCK_REPORT });
  });
}

export const LONG_CV =
  "Jane Okafor, senior backend engineer with six years of experience building Python services. ".repeat(
    3
  );
export const LONG_JOB =
  "Senior platform engineer role requiring four years of experience with Python, Kubernetes and cloud infrastructure. ".repeat(
    3
  );
