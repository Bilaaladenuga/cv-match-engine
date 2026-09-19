"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Trash2, Clock, ArrowRight, CheckCircle2, XCircle } from "lucide-react";
import { listAnalyses, deleteAnalysis, clearHistory } from "@/lib/history";
import type { StoredAnalysis } from "@/lib/history";

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
    <main className="bg-canvas min-h-screen">
      {/* Nav */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-center py-6">
        <div className="flex items-center gap-8 rounded-pill bg-paper px-4 py-3 sm:px-8">
          <Link href="/" className="font-display text-xl uppercase tracking-tight text-carbon">
            CV Match
          </Link>
          <div className="flex items-center gap-6">
            <Link href="/analyze" className="text-body font-medium text-slate hover:text-carbon transition-colors">
              Analyze
            </Link>
            <Link href="/history" className="text-body font-medium text-carbon">
              History
            </Link>
          </div>
        </div>
      </nav>

      <div className="max-w-page mx-auto px-4 sm:px-8 pt-32 pb-section">
        {/* Header */}
        <div className="mb-12 flex items-end justify-between">
          <div>
            <span className="tag mb-4 inline-block">HISTORY</span>
            <h1 className="heading-display text-display text-carbon">
              Past Analyses
            </h1>
            <p className="text-body text-slate mt-2">
              Saved in this browser only. Clear your browser data and they are gone.
            </p>
          </div>
          <Link href="/analyze" className="btn-primary">
            New Analysis
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </div>

        {entries === null ? null : entries.length === 0 ? (
          <div className="card p-16 text-center">
            <div className="mx-auto mb-4 h-14 w-14 rounded-card bg-ash/30 flex items-center justify-center">
              <Clock className="h-6 w-6 text-smoke" />
            </div>
            <p className="font-body text-sub font-medium uppercase text-carbon">
              No analyses yet
            </p>
            <p className="text-body text-slate mt-2 mb-6">
              Every analysis you run on this device shows up here.
            </p>
            <Link href="/analyze" className="btn-primary inline-flex">
              Run Your First Analysis
            </Link>
          </div>
        ) : (
          <>
            <div className="mb-4 flex justify-end">
              {confirmingClear ? (
                <span className="flex items-center gap-3 label-mono text-smoke">
                  Delete all {entries.length} analyses?
                  <button
                    onClick={() => {
                      clearHistory();
                      setEntries([]);
                      setConfirmingClear(false);
                    }}
                    className="font-medium text-carbon hover:underline"
                  >
                    Yes, delete all
                  </button>
                  <button
                    onClick={() => setConfirmingClear(false)}
                    className="hover:text-carbon"
                  >
                    Cancel
                  </button>
                </span>
              ) : (
                <button
                  onClick={() => setConfirmingClear(true)}
                  className="flex items-center gap-1.5 label-mono text-smoke hover:text-carbon transition-colors"
                >
                  <Trash2 className="h-3 w-3" />
                  Clear history
                </button>
              )}
            </div>

            <div className="card overflow-hidden divide-y divide-ash/30">
              {entries.map((e) => (
                <div
                  key={e.id}
                  className="group flex items-center justify-between px-6 py-5 transition-colors hover:bg-mist"
                >
                  <Link href={`/history/${e.id}`} className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate font-medium text-carbon group-hover:underline">
                          {e.title}
                        </p>
                        <p className="label-mono mt-1">{fmtDate(e.created_at)}</p>
                        {e.report.skill_evidence && e.report.skill_evidence.length > 0 ? (
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            {e.report.skill_evidence
                              .filter((s) => s.status === "matched")
                              .slice(0, 3)
                              .map((s) => (
                                <span
                                  key={s.skill}
                                  className="inline-flex items-center gap-1 rounded-tag bg-mint px-3 py-0.5 text-caption font-mono text-carbon"
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
                                  className="inline-flex items-center gap-1 rounded-tag bg-carbon px-3 py-0.5 text-caption font-mono text-paper"
                                >
                                  <XCircle className="h-2.5 w-2.5" />
                                  {s.skill}
                                </span>
                              ))}
                            {e.report.skill_evidence.filter((s) => s.status === "matched").length >
                              3 ? (
                              <span className="label-mono">
                                +{e.report.skill_evidence.filter((s) => s.status === "matched").length - 3} more
                              </span>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="tag">{e.report.band}</span>
                        <span className="font-display text-heading text-carbon">
                          {e.report.overall_percent ?? pct(e.report.overall_score)}
                          <span className="label-mono text-smoke">/100</span>
                        </span>
                      </div>
                    </div>
                  </Link>
                  <button
                    onClick={() => remove(e.id)}
                    aria-label={`Delete: ${e.title}`}
                    className="ml-4 flex-shrink-0 rounded-lg p-2 text-ash opacity-0 transition-all group-hover:opacity-100 hover:bg-carbon hover:text-paper"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </main>
  );
}
