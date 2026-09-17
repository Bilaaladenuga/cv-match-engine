"use client";

/**
 * /history — analyses saved in THIS browser (localStorage; no accounts
 * by design, the server never stores analyses). Rows link to the full
 * saved report at /history/[id].
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { listAnalyses, deleteAnalysis, clearHistory } from "@/lib/history";
import type { StoredAnalysis } from "@/lib/history";
import { BandBadge, Card } from "@/components/ui";

function fmtDate(iso: string): string {
  const d = new Date(iso);
  return isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
}

function pct(score: number): number {
  return Math.round((score ?? 0) * 100);
}

export default function HistoryPage() {
  // null = not hydrated yet (avoids SSR/localStorage mismatch)
  const [entries, setEntries] = useState<StoredAnalysis[] | null>(null);
  const [confirmingClear, setConfirmingClear] = useState(false);

  useEffect(() => {
    setEntries(listAnalyses());
  }, []);

  function remove(id: string) {
    deleteAnalysis(id);
    setEntries(listAnalyses());
  }

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">
            Analysis history
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Saved in this browser only — clearing your browser data removes
            them. The server keeps nothing.
          </p>
        </div>
        <Link
          href="/analyze"
          className="text-sm text-gray-500 hover:text-gray-900"
        >
          New analysis →
        </Link>
      </div>

      {entries === null ? null : entries.length === 0 ? (
        <Card className="px-6 py-10 text-center">
          <p className="text-sm text-gray-500">
            No saved analyses yet — every analysis you run on this device is
            saved here automatically.
          </p>
          <Link
            href="/analyze"
            className="mt-4 inline-block rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white hover:bg-gray-800"
          >
            Run an analysis
          </Link>
        </Card>
      ) : (
        <>
          <div className="mb-3 flex justify-end">
            {confirmingClear ? (
              <span className="flex items-center gap-3 text-xs text-gray-500">
                Delete all {entries.length} saved analyses?
                <button
                  onClick={() => {
                    clearHistory();
                    setEntries([]);
                    setConfirmingClear(false);
                  }}
                  className="font-medium text-red-600 hover:text-red-700"
                >
                  Yes, delete
                </button>
                <button
                  onClick={() => setConfirmingClear(false)}
                  className="hover:text-gray-700"
                >
                  Cancel
                </button>
              </span>
            ) : (
              <button
                onClick={() => setConfirmingClear(true)}
                className="text-xs text-gray-400 hover:text-gray-600"
              >
                Clear history
              </button>
            )}
          </div>
          <Card className="divide-y divide-gray-100">
            {entries.map((e) => (
              <div
                key={e.id}
                className="group flex items-center justify-between px-5 py-4 hover:bg-gray-50"
              >
                <Link href={`/history/${e.id}`} className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="min-w-0">
                      <p className="truncate font-medium text-gray-900">
                        {e.title}
                      </p>
                      <p className="mt-0.5 text-xs text-gray-400">
                        {fmtDate(e.created_at)} · model {e.report.model_version}
                        {e.report.ml_details?.label
                          ? ` · ML: ${e.report.ml_details.label}`
                          : ""}
                      </p>
                    </div>
                    <div className="flex items-center gap-3">
                      <BandBadge band={e.report.band} />
                      <span className="text-sm font-semibold tabular-nums text-gray-900">
                        {e.report.overall_percent ?? pct(e.report.overall_score)}
                        <span className="text-xs font-normal text-gray-400">
                          /100
                        </span>
                      </span>
                    </div>
                  </div>
                </Link>
                <button
                  onClick={() => remove(e.id)}
                  aria-label={`Delete analysis: ${e.title}`}
                  className="ml-3 flex-shrink-0 text-xs text-gray-300 hover:text-red-600"
                >
                  ✕
                </button>
              </div>
            ))}
          </Card>
        </>
      )}
    </main>
  );
}
