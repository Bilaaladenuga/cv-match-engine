import clsx from "clsx";
import { AlertTriangle } from "lucide-react";
import type { ReactNode } from "react";

export function Card({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={clsx("rounded-card bg-paper p-6", className)}>
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
}: {
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="border-b border-ash/30 px-6 py-4">
      <h3 className="font-body text-sub font-medium uppercase text-carbon">
        {title}
      </h3>
      {subtitle ? (
        <p className="mt-1 label-mono text-smoke">{subtitle}</p>
      ) : null}
    </div>
  );
}

const bandStyles: Record<string, string> = {
  "Excellent match": "bg-mint text-carbon",
  "Good match": "bg-paper text-carbon border border-ash",
  "Moderate match": "bg-ash/30 text-carbon",
  "Fair match": "bg-graphite text-paper",
  "Weak match": "bg-carbon text-paper",
};

export function BandBadge({ band }: { band: string | null | undefined }) {
  if (!band) return null;
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-tag px-4 py-1 text-caption font-mono font-medium",
        bandStyles[band] ?? "bg-ash/30 text-carbon"
      )}
    >
      {band}
    </span>
  );
}

export function ScoreBar({
  label,
  value,
  max = 1,
  color = "bg-carbon",
}: {
  label: string;
  value: number;
  max?: number;
  color?: string;
}) {
  const pct = Math.round((value / max) * 100);
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="label-mono text-smoke">{label}</span>
        <span className="font-mono text-caption text-carbon">{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-ash/30 overflow-hidden">
        <div
          className={clsx("h-full rounded-full transition-all", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function ScoreDial({
  score,
  size = 120,
}: {
  score: number;
  size?: number;
}) {
  const pct = Math.round(score * 100);
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="rotate-[-90deg]">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#c6c6c6"
          strokeWidth="6"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#000000"
          strokeWidth="6"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-700"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-display text-heading text-carbon">{pct}</span>
        <span className="label-mono text-smoke">/100</span>
      </div>
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 rounded-card bg-voltage/20 border border-voltage p-4">
      <AlertTriangle className="h-5 w-5 text-carbon shrink-0 mt-0.5" />
      <div>
        <p className="font-body text-sub font-medium text-carbon">Something went wrong</p>
        <p className="mt-1 text-body text-slate">{message}</p>
      </div>
    </div>
  );
}

export function Loading({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-ash border-t-carbon mb-4" />
      {label ? <p className="label-mono text-smoke">{label}</p> : null}
    </div>
  );
}

export function MatchReportSkeleton() {
  return (
    <div className="card p-8 animate-pulse">
      <div className="flex items-center gap-6 mb-8">
        <div className="h-24 w-24 rounded-full bg-ash/30" />
        <div className="flex-1">
          <div className="h-6 w-48 rounded bg-ash/30 mb-2" />
          <div className="h-4 w-32 rounded bg-ash/30" />
        </div>
      </div>
      <div className="space-y-4">
        <div className="h-4 w-full rounded bg-ash/30" />
        <div className="h-4 w-3/4 rounded bg-ash/30" />
        <div className="h-4 w-5/6 rounded bg-ash/30" />
        <div className="h-4 w-2/3 rounded bg-ash/30" />
      </div>
    </div>
  );
}
