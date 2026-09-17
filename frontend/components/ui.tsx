/**
 * Shared UI primitives — premium design system.
 *
 * Liquid glass aesthetic with gradient accents, layered shadows,
 * and smooth micro-interactions. Every component is accessible
 * and respects prefers-reduced-motion.
 */

import clsx from "clsx";
import { AlertTriangle } from "lucide-react";
import type { ReactNode } from "react";

// ---------------------------------------------------------------------------
// Card
// ---------------------------------------------------------------------------

export function Card({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={clsx(
        "rounded-2xl border border-white/60 bg-white/80 shadow-premium backdrop-blur-sm",
        className
      )}
    >
      {children}
    </div>
  );
}

export function GlassCard({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={clsx(
        "rounded-2xl border border-white/40 bg-white/60 shadow-glass backdrop-blur-md",
        className
      )}
    >
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
    <div className="border-b border-gray-100/80 px-6 py-4">
      <h3 className="font-heading text-sm font-semibold tracking-tight text-gray-900">
        {title}
      </h3>
      {subtitle ? (
        <p className="mt-0.5 text-xs text-gray-500">{subtitle}</p>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Band Badge
// ---------------------------------------------------------------------------

const bandStyles: Record<string, string> = {
  "Excellent match":
    "bg-gradient-to-r from-emerald-50 to-green-50 text-emerald-700 border-emerald-200/80 shadow-[0_0_12px_rgba(5,150,105,0.1)]",
  "Good match":
    "bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 border-blue-200/80 shadow-[0_0_12px_rgba(37,99,235,0.1)]",
  "Moderate match":
    "bg-gradient-to-r from-amber-50 to-orange-50 text-amber-700 border-amber-200/80",
  "Fair match":
    "bg-gradient-to-r from-orange-50 to-red-50 text-orange-700 border-orange-200/80",
  "Weak match":
    "bg-gradient-to-r from-red-50 to-rose-50 text-red-700 border-red-200/80",
};

export function BandBadge({ band }: { band: string | null | undefined }) {
  if (!band) return null;
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full border px-3 py-0.5 text-xs font-semibold tracking-wide",
        bandStyles[band] ?? "bg-gray-50 text-gray-600 border-gray-200"
      )}
    >
      {band}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Score Dial — animated gradient ring
// ---------------------------------------------------------------------------

export function ScoreDial({
  percent,
  size = 140,
}: {
  percent: number;
  size?: number;
}) {
  const stroke = 10;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(100, percent));
  const offset = circumference * (1 - clamped / 100);

  const gradientId = `score-grad-${size}`;
  const glowId = `score-glow-${size}`;

  return (
    <div className="relative inline-flex" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#2563EB" />
            <stop offset="50%" stopColor="#059669" />
            <stop offset="100%" stopColor="#10B981" />
          </linearGradient>
          <filter id={glowId}>
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#E2E8F0"
          strokeWidth={stroke}
          opacity={0.5}
        />
        {/* Score arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={stroke}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          filter={`url(#${glowId})`}
          style={{ transition: "stroke-dashoffset 1s ease-out" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-heading text-3xl font-bold tabular-nums text-gray-900">
          {clamped}
        </span>
        <span className="text-[10px] font-medium uppercase tracking-widest text-gray-400">
          / 100
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Score Bar — gradient fill with animated width
// ---------------------------------------------------------------------------

export function ScoreBar({
  label,
  value,
  weight,
  evidence,
}: {
  label: string;
  value: number;
  weight?: number;
  evidence?: string;
}) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);

  const barColor =
    pct >= 70
      ? "from-blue-600 to-accent-500"
      : pct >= 50
        ? "from-blue-600 to-blue-400"
        : pct >= 30
          ? "from-amber-500 to-orange-400"
          : "from-red-500 to-red-400";

  return (
    <div>
      <div className="flex items-baseline justify-between text-sm">
        <span className="font-medium text-gray-700">
          {label}
          {weight !== undefined ? (
            <span className="ml-1.5 text-xs text-gray-400">
              weight {Math.round(weight * 100)}%
            </span>
          ) : null}
        </span>
        <span className="tabular-nums font-semibold text-gray-900">{pct}</span>
      </div>
      <div className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-gray-100/80">
        <div
          className={clsx(
            "h-full rounded-full bg-gradient-to-r transition-all duration-700 ease-out",
            barColor
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      {evidence ? (
        <p className="mt-1 text-xs text-gray-500">{evidence}</p>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Loading — shimmer skeleton or spinner
// ---------------------------------------------------------------------------

export function Loading({ label = "Analyzing…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3">
      <div className="relative">
        <span className="block h-5 w-5 animate-spin rounded-full border-2 border-primary-200 border-t-primary-600" />
        <span className="absolute inset-0 block h-5 w-5 animate-ping rounded-full border border-primary-300 opacity-30" />
      </div>
      <span className="text-sm font-medium text-gray-600">{label}</span>
    </div>
  );
}

export function ShimmerCard({ lines = 3 }: { lines?: number }) {
  return (
    <div className="card-premium p-6 space-y-3">
      <div className="h-4 w-1/3 rounded shimmer-bg" />
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="h-3 w-full rounded shimmer-bg" style={{ width: `${70 + Math.random() * 30}%` }} />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Error Note
// ---------------------------------------------------------------------------

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-red-200/80 bg-gradient-to-r from-red-50 to-rose-50 px-5 py-4 text-sm text-red-700 shadow-sm">
      <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-red-100">
        <AlertTriangle className="h-3.5 w-3.5 text-red-600" />
      </div>
      <span>{message}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Match Report Skeleton — shimmer placeholder while the API responds
// ---------------------------------------------------------------------------

export function MatchReportSkeleton() {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Score dial skeleton */}
      <Card className="overflow-hidden">
        <div className="flex flex-col items-center gap-8 px-8 py-8 sm:flex-row">
          <div className="relative h-[140px] w-[140px] shrink-0">
            <div className="h-full w-full rounded-full shimmer-bg" />
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="h-8 w-12 rounded shimmer-bg" />
              <div className="mt-1 h-2 w-8 rounded shimmer-bg" />
            </div>
          </div>
          <div className="flex-1 space-y-3">
            <div className="h-6 w-32 rounded shimmer-bg" />
            <div className="h-4 w-64 rounded shimmer-bg" />
          </div>
        </div>
      </Card>

      {/* Score breakdown skeleton */}
      <Card>
        <div className="border-b border-gray-100/60 px-6 py-4">
          <div className="h-4 w-40 rounded shimmer-bg" />
          <div className="mt-1 h-3 w-56 rounded shimmer-bg" />
        </div>
        <div className="space-y-5 px-6 py-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i}>
              <div className="flex items-baseline justify-between">
                <div className="h-3 w-24 rounded shimmer-bg" />
                <div className="h-3 w-8 rounded shimmer-bg" />
              </div>
              <div className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-gray-100/80">
                <div
                  className="h-full rounded-full shimmer-bg"
                  style={{ width: `${50 + Math.random() * 40}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Skill table skeleton */}
      <Card>
        <div className="border-b border-gray-100/60 px-6 py-4">
          <div className="h-4 w-36 rounded shimmer-bg" />
        </div>
        <div className="px-5 py-4">
          <div className="space-y-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <div
                key={i}
                className="flex items-center gap-4 border-b border-gray-50 py-3 last:border-0"
              >
                <div className="h-4 w-4 shrink-0 rounded-full shimmer-bg" />
                <div className="h-3 w-28 rounded shimmer-bg" />
                <div className="h-3 w-20 rounded shimmer-bg" />
                <div className="h-5 w-24 rounded-full shimmer-bg" />
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Factors + Recommendations skeleton */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <div className="border-b border-gray-100/60 px-6 py-4">
            <div className="h-4 w-32 rounded shimmer-bg" />
          </div>
          <div className="space-y-3 px-5 py-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="flex items-start gap-3">
                <div className="mt-0.5 h-4 w-4 shrink-0 rounded shimmer-bg" />
                <div className="h-3 flex-1 rounded shimmer-bg" />
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <div className="border-b border-gray-100/60 px-6 py-4">
            <div className="h-4 w-32 rounded shimmer-bg" />
          </div>
          <div className="space-y-3 px-5 py-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="flex items-start gap-3">
                <div className="mt-0.5 h-4 w-4 shrink-0 rounded shimmer-bg" />
                <div className="h-3 flex-1 rounded shimmer-bg" />
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
