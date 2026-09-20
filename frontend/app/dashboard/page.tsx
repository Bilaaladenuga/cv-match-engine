"use client";

/**
 * /dashboard — overview of analyses saved in THIS browser (localStorage;
 * no accounts by design). Stats are computed client-side from the local
 * history — no server round-trip.
 */

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import MobileNav from "@/components/MobileNav";
import {
  BarChart3,
  TrendingUp,
  Award,
  ArrowRight,
  LayoutDashboard,
} from "lucide-react";
import { listAnalyses } from "@/lib/history";
import type { StoredAnalysis } from "@/lib/history";
import { BandBadge, Card, CardHeader, ScoreBar } from "@/components/ui";

function Stat({
  label,
  value,
  hint,
  icon,
}: {
  label: string;
  value: string;
  hint?: string;
  icon: React.ReactNode;
}) {
  return (
    <Card>
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-carbon text-paper">
          {icon}
        </div>
        <div className="min-w-0">
          <p className="label-mono">{label}</p>
          <p className="font-display text-heading-sm text-carbon tabular-nums">
            {value}
          </p>
        </div>
      </div>
      {hint ? <p className="mt-2 label-mono">{hint}</p> : null}
    </Card>
  );
}

export default function DashboardPage() {
  const [entries, setEntries] = useState<StoredAnalysis[] | null>(null);

  useEffect(() => {
    setEntries(listAnalyses());
  }, []);

  const stats = useMemo(() => {
    if (!entries || entries.length === 0) return null;
    const scores = entries.map(
      (e) => e.report.overall_percent ?? Math.round(e.report.overall_score * 100)
    );
    const avg = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
    const bands = entries.reduce<Record<string, number>>((acc, e) => {
      if (e.report.band) acc[e.report.band] = (acc[e.report.band] ?? 0) + 1;
      return acc;
    }, {});
    return { count: entries.length, avg, bands };
  }, [entries]);

  return (
    <main className="bg-canvas min-h-screen">
      {/* Nav */}
      <MobileNav />
      <nav className="fixed top-0 left-0 right-0 z-50 hidden items-center justify-center px-3 py-6 md:flex">
        <div className="flex items-center gap-4 rounded-pill bg-paper px-4 py-3 sm:gap-8 sm:px-8">
          <Link
            href="/"
            className="font-display text-lg uppercase tracking-tight text-carbon sm:text-xl"
          >
            CV Match
          </Link>
          <div className="flex items-center gap-4 sm:gap-6">
            <Link
              href="/analyze"
              className="text-body-sm font-medium text-slate hover:text-carbon transition-colors sm:text-body"
            >
              Analyze
            </Link>
            <Link
              href="/dashboard"
              className="text-body-sm font-medium text-carbon sm:text-body"
            >
              Dashboard
            </Link>
            <Link
              href="/history"
              className="text-body-sm font-medium text-slate hover:text-carbon transition-colors sm:text-body"
            >
              History
            </Link>
          </div>
        </div>
      </nav>

      <div className="max-w-page mx-auto px-4 sm:px-8 pt-32 pb-section">
        {/* Header */}
        <div className="mb-10 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <span className="tag mb-4 inline-block">DASHBOARD</span>
            <h1 className="heading-display text-display text-carbon">
              Overview
            </h1>
            <p className="text-body text-slate mt-2">
              Analyses saved in this browser.
            </p>
          </div>
          <Link href="/analyze" className="btn-primary shrink-0 self-start sm:self-auto">
            New analysis
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </div>

        {entries === null ? null : stats ? (
          <>
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
              <Stat
                label="Analyses"
                value={String(stats.count)}
                hint="saved in this browser"
                icon={<BarChart3 className="h-5 w-5" />}
              />
              <Stat
                label="Average score"
                value={`${stats.avg}`}
                hint="across saved analyses"
                icon={<TrendingUp className="h-5 w-5" />}
              />
              <Stat
                label="Good or better"
                value={String(
                  Object.entries(stats.bands)
                    .filter(
                      ([band]) =>
                        band === "Good match" || band === "Excellent match"
                    )
                    .reduce((a, [, n]) => a + n, 0)
                )}
                hint="analyses scoring 70+"
                icon={<Award className="h-5 w-5" />}
              />
            </div>

            <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-3">
              <Card className="lg:col-span-2 overflow-hidden">
                <CardHeader title="Recent analyses" />
                <div className="divide-y divide-ash/30">
                  {entries.slice(0, 5).map((e) => (
                    <Link
                      key={e.id}
                      href={`/history/${e.id}`}
                      className="flex items-center justify-between gap-3 px-4 py-3.5 transition-colors hover:bg-mist sm:px-5"
                    >
                      <div className="min-w-0">
                        <p className="truncate text-body-sm font-medium text-carbon hover:underline">
                          {e.title}
                        </p>
                        <p className="label-mono mt-0.5">
                          {new Date(e.created_at).toLocaleString(undefined, {
                            dateStyle: "medium",
                            timeStyle: "short",
                          })}
                        </p>
                      </div>
                      <div className="flex shrink-0 items-center gap-3">
                        <BandBadge band={e.report.band} />
                        <span className="font-display text-heading-sm text-carbon tabular-nums">
                          {e.report.overall_percent}
                        </span>
                      </div>
                    </Link>
                  ))}
                </div>
              </Card>

              <Card className="overflow-hidden">
                <CardHeader title="Score distribution" />
                <div className="space-y-4 px-4 py-4 sm:px-5">
                  {Object.entries(stats.bands)
                    .sort((a, b) => b[1] - a[1])
                    .map(([band, n]) => {
                      const maxCount = Math.max(...Object.values(stats.bands));
                      const pctValue = Math.round((n / maxCount) * 100);
                      return (
                        <div key={band}>
                          <div className="mb-1 flex items-center justify-between">
                            <BandBadge band={band} />
                            <span className="font-mono text-caption font-medium text-carbon tabular-nums">
                              {n}
                            </span>
                          </div>
                          <ScoreBar
                            label=""
                            value={pctValue}
                            color="bg-carbon"
                          />
                        </div>
                      );
                    })}
                </div>
              </Card>
            </div>
          </>
        ) : (
          <Card className="px-6 py-14 text-center">
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-card bg-ash/30">
              <LayoutDashboard className="h-6 w-6 text-smoke" />
            </div>
            <p className="font-body text-sub font-medium uppercase text-carbon">
              Nothing to summarize yet
            </p>
            <p className="mt-1 text-body-sm text-slate">
              Run your first analysis to see stats here.
            </p>
            <Link
              href="/analyze"
              className="btn-primary mt-6 inline-flex items-center gap-2"
            >
              Analyze a CV
              <ArrowRight className="h-4 w-4" />
            </Link>
          </Card>
        )}
      </div>
    </main>
  );
}
