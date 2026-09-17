"use client";

/**
 * /history/[match_id] — full stored report (GET /api/matches/{id}).
 * The stored shape is mapped into the same display contract the live
 * /analyze report uses, so both views render identically.
 */

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { getMatchDetail, apiErrorMessage } from "@/lib/api";
import type { MatchDetail } from "@/lib/types";
import { ErrorNote, Loading } from "@/components/ui";
import { MatchReportView } from "@/components/MatchReportView";

export default function MatchDetailPage({
  params,
}: {
  params: Promise<{ match_id: string }>;
}) {
  const { match_id } = use(params);
  const [detail, setDetail] = useState<MatchDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMatchDetail(Number(match_id))
      .then(setDetail)
      .catch((err) => setError(apiErrorMessage(err)));
  }, [match_id]);

  if (error) {
    return (
      <main className="mx-auto max-w-4xl px-6 py-10">
        <ErrorNote message={error} />
        <Link
          href="/history"
          className="mt-4 inline-block text-sm text-gray-500 hover:text-gray-900"
        >
          ← Back to history
        </Link>
      </main>
    );
  }

  if (!detail) {
    return (
      <main className="mx-auto max-w-4xl px-6 py-10">
        <Loading label="Loading report…" />
      </main>
    );
  }

  // Map the stored schema onto the live-report display contract.
  const componentNames = ["skills", "semantic", "experience", "education"] as const;
  const components = componentNames
    .filter((n) => detail.component_scores?.[n] !== undefined)
    .map((n) => ({
      name: n,
      raw_score: detail.component_scores[n],
      weight: 0,
      weighted: detail.component_scores[n],
      evidence: "",
    }));

  const title =
    detail.candidate_name ?? `Candidate #${detail.candidate_id}`;
  const subtitle = [
    detail.job_title ?? `Job #${detail.job_id}`,
    detail.job_company,
    detail.created_at
      ? new Date(detail.created_at).toLocaleString(undefined, {
          dateStyle: "medium",
          timeStyle: "short",
        })
      : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <Link
        href="/history"
        className="mb-6 inline-block text-sm text-gray-500 hover:text-gray-900"
      >
        ← Back to history
      </Link>
      <MatchReportView
        report={{
          model_version: detail.model_version,
          overall_score: detail.overall_score,
          overall_percent: Math.round(detail.overall_score * 100),
          band: detail.band ?? "",
          components,
          weights: {},
          positive_factors: detail.explanation.positive_factors,
          negative_factors: detail.explanation.negative_factors,
          recommendations: detail.explanation.recommendations,
          disclaimer:
            "This score is a model-estimated compatibility between the CV and the " +
            "job description. It is a decision-support signal, not a prediction of " +
            "hiring outcomes and not a substitute for human judgement.",
          skill_evidence: undefined, // stored reports keep the three lists
          ml_details: {
            label: detail.ml.label ?? undefined,
            fit_score: detail.ml.fit_score ?? undefined,
            probabilities: detail.ml.probabilities ?? undefined,
            calibration_method: detail.ml.calibration_method,
          },
        }}
        lists={{
          matched: detail.explanation.matched_skills,
          partial: detail.explanation.partial_skills,
          missing: detail.explanation.missing_skills,
        }}
        title={`${title} — stored analysis`}
        subtitle={subtitle}
      />
    </main>
  );
}
