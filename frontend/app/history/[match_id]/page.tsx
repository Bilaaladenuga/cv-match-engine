"use client";

/**
 * /history/[id] — full saved report from browser-localStorage history.
 * Renders the exact MatchReport the live /analyze page shows, so both
 * views are identical.
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { getAnalysis } from "@/lib/history";
import type { StoredAnalysis } from "@/lib/history";
import { Loading } from "@/components/ui";
import { MatchReportView } from "@/components/MatchReportView";

// Next 14 passes params to client components as a plain object
// (Promise-based params are Next 15+) — do NOT unwrap with use().
export default function SavedMatchPage({
  params,
}: {
  params: { id: string };
}) {
  const id = params.id;
  const [entry, setEntry] = useState<StoredAnalysis | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    const found = getAnalysis(id);
    if (found) setEntry(found);
    else setMissing(true);
  }, [id]);

  if (missing) {
    return (
      <main className="mx-auto max-w-4xl px-6 py-10">
        <p className="text-sm text-gray-500">
          This analysis isn&apos;t in this browser&apos;s saved history — it
          may have been deleted, run on another device, or the link is stale.
        </p>
        <Link
          href="/history"
          className="mt-4 inline-block text-sm text-gray-500 hover:text-gray-900"
        >
          ← Back to history
        </Link>
      </main>
    );
  }

  if (!entry) {
    return (
      <main className="mx-auto max-w-4xl px-6 py-10">
        <Loading label="Loading saved report…" />
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <Link
        href="/history"
        className="mb-6 inline-block text-sm text-gray-500 hover:text-gray-900"
      >
        ← Back to history
      </Link>
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
