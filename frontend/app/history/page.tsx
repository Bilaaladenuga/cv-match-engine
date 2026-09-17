"use client";

/**
 * /history — analyses saved in THIS browser (localStorage; no accounts
 * by design, the server never stores analyses). Rows link to the full
 * saved report at /history/[id].
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { Trash2, Clock, ArrowRight, CheckCircle2, XCircle } from "lucide-react";
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
          <h1 className="font-heading text-2xl font-bold tracking-tight text-gray-900">
            Analysis history
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Saved in this browser only — clearing your browser data removes
            them. The server keeps nothing.
          </p>
        </div>
        <Link
          href="/analyze"
          className="flex items-center gap-1.5 text-sm font-medium text-gray-500 transition-colors hover:text-primary-600"
        >
          New analysis <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>

      {entries === null ? null : entries.length === 0 ? (
        <Card className="px-8 py-14 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-600/8">
            <Clock className="h-6 w-6 text-primary-600" />
          </div>
          <p className="font-heading text-base font-semibold text-gray-900">
            No saved analyses yet
          </p>
          <p className="mt-1 text-sm text-gray-500">
            Every analysis you run on this device is saved here automatically.
          </p>
          <Link
            href="/analyze"
            className="btn-primary mt-6 inline-flex"
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
                  className="font-medium text-red-600 transition-colors hover:text-red-700"
                >
                  Yes, delete
                </button>
                <button
                  onClick={() => setConfirmingClear(false)}
                  className="transition-colors hover:text-gray-700"
                >
                  Cancel
                </button>
              </span>
            ) : (
              <button
                onClick={() => setConfirmingClear(true)}
                className="flex items-center gap-1.5 text-xs font-medium text-gray-400 transition-colors hover:text-red-500"
              >
                <Trash2 className="h-3 w-3" />
                Clear history
              </button>
            )}
          </div>
          <Card className="divide-y divide-gray-100/60 overflow-hidden">
            {entries.map((e) => (
              <div
                key={e.id}
                className="group flex items-center justify-between px-5 py-4 transition-colors hover:bg-gray-50/50"
              >
                <Link href={`/history/${e.id}`} className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="min-w-0">
                      <p className="truncate font-medium text-gray-900 group-hover:text-primary-700 transition-colors">
                        {e.title}
                      </p>
                      <p className="mt-0.5 text-xs text-gray-400">
                        {fmtDate(e.created_at)}
                      </p>
                      {/* Quick skill preview */}
                      {e.report.skill_evidence && e.report.skill_evidence.length > 0 ? (
                        <div className="mt-1.5 flex flex-wrap gap-1.5">
                          {e.report.skill_evidence
                            .filter((s) => s.status === "matched")
                            .slice(0, 3)
                            .map((s) => (
                              <span
                                key={s.skill}
                                className="inline-flex items-center gap-0.5 rounded-full bg-accent-50 px-2 py-0.5 text-[10px] font-medium text-accent-700"
                              >
                                <CheckCircle2 className="h-2.5 w-2.5" />
                                {s.skill}
                              </span>
                            ))}
                          {e.report.skill_evidence
                            .filter((s) => s.status === "missing")
                            .slice(0, 2)
                            .map((s) => (
                              <span
                                key={s.skill}
                                className="inline-flex items-center gap-0.5 rounded-full bg-red-50 px-2 py-0.5 text-[10px] font-medium text-red-600"
                              >
                                <XCircle className="h-2.5 w-2.5" />
                                {s.skill}
                              </span>
                            ))}
                          {e.report.skill_evidence.filter((s) => s.status === "matched").length >
                            3 ? (
                            <span className="text-[10px] text-gray-400">
                              +{e.report.skill_evidence.filter((s) => s.status === "matched").length - 3} more
                            </span>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                    <div className="flex items-center gap-3">
                      <BandBadge band={e.report.band} />
                      <span className="text-sm font-bold tabular-nums text-gray-900">
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
                  className="ml-3 flex-shrink-0 rounded-md p-1.5 text-gray-300 opacity-0 transition-all group-hover:opacity-100 hover:bg-red-50 hover:text-red-500"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </Card>
        </>
      )}
    </main>
  );
}
