"use client";

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
import { Loading } from "@/components/ui";
import { MatchReportView } from "@/components/MatchReportView";
import MobileNav from "@/components/MobileNav";

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
      <main className="bg-canvas min-h-screen">
        <MobileNav />
      <nav className="fixed top-0 left-0 right-0 z-50 hidden items-center justify-center px-3 py-6 md:flex">
          <div className="flex items-center gap-4 rounded-pill bg-paper px-4 py-3 sm:gap-8 sm:px-8">
            <Link href="/" className="font-display text-lg uppercase tracking-tight text-carbon sm:text-xl">
              CV Match
            </Link>
          </div>
        </nav>
        <div className="max-w-page mx-auto px-4 sm:px-8 pt-32 pb-section">
          <div className="card p-8 sm:p-16 text-center">
            <p className="font-body text-sub font-medium uppercase text-carbon">
              Analysis not found
            </p>
            <p className="text-body text-slate mt-2 mb-6">
              This analysis might have been deleted, run on another device, or the link is old.
            </p>
            <Link href="/history" className="btn-ghost inline-flex items-center gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to history
            </Link>
          </div>
        </div>
      </main>
    );
  }

  if (!entry) {
    return (
      <main className="bg-canvas min-h-screen">
        <MobileNav />
      <nav className="fixed top-0 left-0 right-0 z-50 hidden items-center justify-center px-3 py-6 md:flex">
          <div className="flex items-center gap-4 rounded-pill bg-paper px-4 py-3 sm:gap-8 sm:px-8">
            <Link href="/" className="font-display text-lg uppercase tracking-tight text-carbon sm:text-xl">
              CV Match
            </Link>
          </div>
        </nav>
        <div className="max-w-page mx-auto px-4 sm:px-8 pt-32 pb-section">
          <div className="card p-8">
            <Loading label="Loading saved report..." />
          </div>
        </div>
      </main>
    );
  }

  const hasDocs = entry.cv_text || entry.job_text;

  return (
    <main className="bg-canvas min-h-screen">
      <MobileNav />
      <nav className="fixed top-0 left-0 right-0 z-50 hidden items-center justify-center px-3 py-6 md:flex">
        <div className="flex items-center gap-4 rounded-pill bg-paper px-4 py-3 sm:gap-8 sm:px-8">
          <Link href="/" className="font-display text-lg uppercase tracking-tight text-carbon sm:text-xl">
            CV Match
          </Link>
          <div className="flex items-center gap-4 sm:gap-6">
            <Link href="/analyze" className="text-body-sm font-medium text-slate hover:text-carbon transition-colors sm:text-body">
              Analyze
            </Link>
            <Link href="/history" className="text-body-sm font-medium text-slate hover:text-carbon transition-colors sm:text-body">
              History
            </Link>
          </div>
        </div>
      </nav>

      <div className="max-w-page mx-auto px-4 sm:px-8 pt-32 pb-section">
        <Link
          href="/history"
          className="mb-8 inline-flex items-center gap-2 label-mono text-smoke hover:text-carbon transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to history
        </Link>

        {/* Original documents */}
        {hasDocs ? (
          <div className="card mb-8 overflow-hidden">
            <button
              onClick={() => setShowDocs(!showDocs)}
              className="flex w-full items-center justify-between px-6 py-4 text-left transition-colors hover:bg-mist"
            >
              <div className="flex items-center gap-3">
                <div className="h-8 w-8 rounded-lg bg-carbon text-paper flex items-center justify-center">
                  <FileText className="h-4 w-4" />
                </div>
                <div>
                  <p className="font-body text-sub font-medium uppercase text-carbon">
                    Original documents
                  </p>
                  <p className="label-mono text-smoke">
                    The CV and job description from this analysis
                  </p>
                </div>
              </div>
              {showDocs ? (
                <ChevronUp className="h-4 w-4 text-smoke" />
              ) : (
                <ChevronDown className="h-4 w-4 text-smoke" />
              )}
            </button>

            {showDocs ? (
              <div className="grid grid-cols-1 gap-4 border-t border-ash/30 px-6 py-5 sm:grid-cols-2">
                {entry.cv_text ? (
                  <div>
                    <div className="mb-2 flex items-center gap-2">
                      <FileText className="h-3.5 w-3.5 text-carbon" />
                      <span className="label-mono">CV / RESUME</span>
                    </div>
                    <div className="max-h-64 overflow-y-auto rounded-card bg-mist p-4">
                      <pre className="whitespace-pre-wrap font-mono text-caption leading-relaxed text-slate">
                        {entry.cv_text}
                      </pre>
                    </div>
                  </div>
                ) : null}
                {entry.job_text ? (
                  <div>
                    <div className="mb-2 flex items-center gap-2">
                      <Briefcase className="h-3.5 w-3.5 text-carbon" />
                      <span className="label-mono">JOB DESCRIPTION</span>
                    </div>
                    <div className="max-h-64 overflow-y-auto rounded-card bg-mist p-4">
                      <pre className="whitespace-pre-wrap font-mono text-caption leading-relaxed text-slate">
                        {entry.job_text}
                      </pre>
                    </div>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
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
          })}. Stored in this browser only.`}
          exportTexts={
            entry.cv_text && entry.job_text
              ? { cvText: entry.cv_text, jobText: entry.job_text }
              : undefined
          }
        />
      </div>
    </main>
  );
}
