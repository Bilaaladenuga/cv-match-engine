"use client";

/**
 * /history — previously stored analyses (GET /api/history).
 * Rows link to the full stored report at /history/[match_id].
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { getHistory, apiErrorMessage } from "@/lib/api";
import type { HistoryEntry } from "@/lib/types";
import { BandBadge, Card, ErrorNote, Loading } from "@/components/ui";

function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
}

function pct(score: number): number {
  return Math.round(score * 100);
}

export default function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHistory()
      .then(setEntries)
      .catch((err) => setError(apiErrorMessage(err)));
  }, []);

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">
            Analysis history
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Stored match reports, newest first.
          </p>
        </div>
        <Link
          href="/analyze"
          className="text-sm text-gray-500 hover:text-gray-900"
        >
          New analysis →
        </Link>
      </div>

      {error ? <ErrorNote message={error} /> : null}
      {!error && entries === null ? <Loading label="Loading history…" /> : null}

      {entries && entries.length === 0 ? (
        <Card className="px-6 py-10 text-center">
          <p className="text-sm text-gray-500">
            No stored analyses yet. Analyses created while signed out are not
            saved — an account system is planned before persistence is
            enabled in the UI.
          </p>
          <Link
            href="/analyze"
            className="mt-4 inline-block rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white hover:bg-gray-800"
          >
            Run an analysis
          </Link>
        </Card>
      ) : null}

      {entries && entries.length > 0 ? (
        <Card className="divide-y divide-gray-100">
          {entries.map((e) => (
            <Link
              key={e.match_id}
              href={`/history/${e.match_id}`}
              className="block px-5 py-4 hover:bg-gray-50"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <span className="font-medium text-gray-900">
                    {e.candidate_name ?? `Candidate #${e.candidate_id}`}
                  </span>
                  <span className="text-gray-400"> · </span>
                  <span className="text-gray-600">
                    {e.job_title ?? `Job #${e.job_id}`}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <BandBadge band={e.band} />
                  <span className="text-sm font-semibold tabular-nums text-gray-900">
                    {pct(e.overall_score)}
                    <span className="text-xs font-normal text-gray-400">/100</span>
                  </span>
                </div>
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-400">
                <span>{fmtDate(e.created_at)}</span>
                <span>model {e.model_version}</span>
                {e.ml_label ? <span>ML: {e.ml_label}</span> : null}
                <span>
                  {e.matched_skills.length} matched ·{" "}
                  {e.missing_skills.length} missing
                </span>
              </div>
            </Link>
          ))}
        </Card>
      ) : null}
    </main>
  );
}
