"use client";

/**
 * /dashboard — overview of analyses saved in THIS browser (localStorage;
 * no accounts by design). Stats are computed client-side from the local
 * history — no server round-trip.
 */

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { listAnalyses } from "@/lib/history";
import type { StoredAnalysis } from "@/lib/history";
import { BandBadge, Card, CardHeader } from "@/components/ui";

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <Card className="px-5 py-4">
      <p className="text-xs uppercase tracking-wide text-gray-400">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-gray-900">{value}</p>
      {hint ? <p className="mt-0.5 text-xs text-gray-400">{hint}</p> : null}
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
    <main className="mx-auto max-w-6xl px-6 py-10">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Dashboard</h1>
          <p className="mt-1 text-sm text-gray-500">
            Overview of analyses saved in this browser.
          </p>
        </div>
        <Link
          href="/analyze"
          className="rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white hover:bg-gray-800"
        >
          New analysis
        </Link>
      </div>

      {entries === null ? null : stats ? (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Stat
              label="Analyses"
              value={String(stats.count)}
              hint="saved in this browser"
            />
            <Stat
              label="Average score"
              value={`${stats.avg}`}
              hint="across saved analyses"
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
            />
          </div>

          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
            <Card className="lg:col-span-2">
              <CardHeader title="Recent analyses" />
              <div className="divide-y divide-gray-100">
                {entries.slice(0, 5).map((e) => (
                  <Link
                    key={e.id}
                    href={`/history/${e.id}`}
                    className="flex items-center justify-between px-5 py-3 hover:bg-gray-50"
                  >
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-gray-900">
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
                      <span className="text-sm font-semibold tabular-nums">
                        {e.report.overall_percent}
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            </Card>

            <Card>
              <CardHeader title="Score distribution" />
              <div className="space-y-3 px-5 py-4">
                {Object.entries(stats.bands)
                  .sort((a, b) => b[1] - a[1])
                  .map(([band, n]) => (
                    <div
                      key={band}
                      className="flex items-center justify-between"
                    >
                      <BandBadge band={band} />
                      <span className="text-sm tabular-nums text-gray-600">
                        {n}
                      </span>
                    </div>
                  ))}
              </div>
            </Card>
          </div>
        </>
      ) : (
        <Card className="px-6 py-10 text-center">
          <p className="text-sm text-gray-500">
            Nothing to summarize yet — run your first analysis.
          </p>
          <Link
            href="/analyze"
            className="mt-4 inline-block rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white hover:bg-gray-800"
          >
            Analyze a CV
          </Link>
        </Card>
      )}
    </main>
  );
}
