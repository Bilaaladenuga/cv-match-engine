"use client";

/**
 * /history/[id] — full saved report from browser-localStorage history.
 * Renders the exact MatchReport the live /analyze page shows, so both
 * views are identical. Also shows the original CV and JD texts that
 * were submitted, so users can review what they analyzed.
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  FileText,
  Briefcase,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { getAnalysis } from "@/lib/history";
import type { StoredAnalysis } from "@/lib/history";
import { Card, Loading } from "@/components/ui";
import { MatchReportView } from "@/components/MatchReportView";

// Next 14 passes params to client components as a plain object
// (Promise-based params are Next 15+) — do NOT unwrap with use().
export default function SavedMatchPage({
  params,
}: {
  params: { match_id: string };
}) {
  const id = params.match_id;
  const [entry, setEntry] = useState<StoredAnalysis | null>(null);
  const [missing, setMissing] = useState(false);
  const [showDocs, setShowDocs] = useState(false);

  useEffect(() => {
    const found = getAnalysis(id);
    if (found) setEntry(found);
    else setMissing(true);
  }, [id]);

  if (missing) {
    return (
      <main className="mx-auto max-w-4xl px-6 py-10">
        <Card className="px-8 py-14 text-center">
          <p className="font-heading text-base font-semibold text-gray-900">
            Analysis not found
          </p>
          <p className="mt-1 text-sm text-gray-500">
            This analysis isn&apos;t in this browser&apos;s saved history — it
            may have been deleted, run on another device, or the link is stale.
          </p>
          <Link
            href="/history"
            className="btn-ghost mt-6 inline-flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to history
          </Link>
        </Card>
      </main>
    );
  }

  if (!entry) {
    return (
      <main className="mx-auto max-w-4xl px-6 py-10">
        <div className="card-premium p-8">
          <Loading label="Loading saved report…" />
        </div>
      </main>
    );
  }

  const hasDocs = entry.cv_text || entry.job_text;

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <Link
        href="/history"
        className="mb-6 inline-flex items-center gap-1.5 text-sm font-medium text-gray-500 transition-colors hover:text-primary-600"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to history
      </Link>

      {/* Original documents — expandable */}
      {hasDocs ? (
        <Card className="mb-6 overflow-hidden">
          <button
            onClick={() => setShowDocs(!showDocs)}
            className="flex w-full items-center justify-between px-6 py-4 text-left transition-colors hover:bg-gray-50/50"
          >
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600/8 text-primary-700">
                <FileText className="h-4 w-4" />
              </div>
              <div>
                <p className="font-heading text-sm font-semibold text-gray-900">
                  View original documents
                </p>
                <p className="text-xs text-gray-400">
                  The CV and job description you submitted for this analysis
                </p>
              </div>
            </div>
            {showDocs ? (
              <ChevronUp className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            )}
          </button>

          {showDocs ? (
            <div className="grid grid-cols-1 gap-4 border-t border-gray-100/60 px-6 py-5 sm:grid-cols-2">
              {entry.cv_text ? (
                <div>
                  <div className="mb-2 flex items-center gap-2">
                    <FileText className="h-3.5 w-3.5 text-primary-600" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">
                      CV / Resume
                    </span>
                  </div>
                  <div className="max-h-64 overflow-y-auto rounded-lg border border-gray-200/60 bg-gray-50/80 p-4">
                    <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-gray-700">
                      {entry.cv_text}
                    </pre>
                  </div>
                </div>
              ) : null}
              {entry.job_text ? (
                <div>
                  <div className="mb-2 flex items-center gap-2">
                    <Briefcase className="h-3.5 w-3.5 text-accent-600" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">
                      Job Description
                    </span>
                  </div>
                  <div className="max-h-64 overflow-y-auto rounded-lg border border-gray-200/60 bg-gray-50/80 p-4">
                    <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-gray-700">
                      {entry.job_text}
                    </pre>
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}
        </Card>
      ) : null}

      <MatchReportView
        report={entry.report}
        lists={{
          matched: (entry.report.skill_evidence ?? [])
            .filter((e) => e.status === "matched")
            .map((e) => e.skill),
          partial: (entry.report.skill_evidence ?? [])
            .filter((e) => e.status === "partial")
            .map((e) => e.skill),
          missing: (entry.report.skill_evidence ?? [])
            .filter((e) => e.status === "missing")
            .map((e) => e.skill),
        }}
        title={entry.title}
        subtitle={`Saved ${new Date(entry.created_at).toLocaleString(undefined, {
          dateStyle: "medium",
          timeStyle: "short",
        })} · stored in this browser only`}
      />
    </main>
  );
}
