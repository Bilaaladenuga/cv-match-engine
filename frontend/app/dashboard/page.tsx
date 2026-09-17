"use client";

/**
 * /dashboard — overview of analyses saved in THIS browser (localStorage;
 * no accounts by design). Stats are computed client-side from the local
 * history — no server round-trip.
 */

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
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
    <Card className="overflow-hidden">
      <div className="h-1 w-full bg-gradient-to-r from-primary-600 to-accent-500" />
      <div className="px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary-600/8 text-primary-700">
            {icon}
          </div>
          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-gray-400">
              {label}
            </p>
            <p className="font-heading text-2xl font-bold tabular-nums text-gray-900">
              {value}
            </p>
          </div>
        </div>
        {hint ? (
          <p className="mt-1 text-xs text-gray-400">{hint}</p>
        ) : null}
      </div>
    </Card>
  );
}

const bandColorMap: Record<string, string> = {
  "Excellent match": "from-emerald-500 to-accent-600",
  "Good match": "from-blue-500 to-primary-600",
  "Moderate match": "from-amber-500 to-orange-500",
  "Fair match": "from-orange-500 to-red-400",
  "Weak match": "from-red-500 to-red-600",
};

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
    <main className="mx-auto max-w-6xl px-6 py-10">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="font-heading text-2xl font-bold tracking-tight text-gray-900">
            Dashboard
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Overview of analyses saved in this browser.
          </p>
        </div>
        <Link href="/analyze" className="btn-primary inline-flex items-center gap-2">
          New analysis
          <ArrowRight className="h-4 w-4" />
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
              <div className="divide-y divide-gray-100/60">
                {entries.slice(0, 5).map((e) => (
                  <Link
                    key={e.id}
                    href={`/history/${e.id}`}
                    className="flex items-center justify-between px-5 py-3.5 transition-colors hover:bg-gray-50/50"
                  >
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-gray-900 hover:text-primary-700 transition-colors">
                        {e.title}
                      </p>
                      <p className="text-xs text-gray-400">
                        {new Date(e.created_at).toLocaleString(undefined, {
                          dateStyle: "medium",
                          timeStyle: "short",
                        })}
                      </p>
                    </div>
                    <div className="flex flex-shrink-0 items-center gap-3">
                      <BandBadge band={e.report.band} />
                      <span className="text-sm font-bold tabular-nums">
                        {e.report.overall_percent}
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            </Card>

            <Card className="overflow-hidden">
              <CardHeader title="Score distribution" />
              <div className="space-y-3 px-5 py-4">
                {Object.entries(stats.bands)
                  .sort((a, b) => b[1] - a[1])
                  .map(([band, n]) => {
                    const maxCount = Math.max(...Object.values(stats.bands));
                    const pctWidth = Math.round((n / maxCount) * 100);
                    return (
                      <div key={band} className="space-y-1">
                        <div className="flex items-center justify-between">
                          <BandBadge band={band} />
                          <span className="text-sm font-semibold tabular-nums text-gray-600">
                            {n}
                          </span>
                        </div>
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
                          <div
                            className={`h-full rounded-full bg-gradient-to-r ${bandColorMap[band] ?? "from-gray-400 to-gray-500"} transition-all duration-500`}
                            style={{ width: `${pctWidth}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </Card>
          </div>
        </>
      ) : (
        <Card className="px-8 py-14 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-600/8">
            <LayoutDashboard className="h-6 w-6 text-primary-600" />
          </div>
          <p className="font-heading text-base font-semibold text-gray-900">
            Nothing to summarize yet
          </p>
          <p className="mt-1 text-sm text-gray-500">
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
    </main>
  );
}
