/**
 * Small shared UI primitives. Deliberately plain: professional SaaS
 * styling (grays, subtle borders) — no gradients or glow effects.
 */

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
    <div
      className={clsx(
        "rounded-lg border border-gray-200 bg-white shadow-sm",
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
    <div className="border-b border-gray-100 px-5 py-3">
      <h3 className="text-sm font-semibold text-gray-900">{title}</h3>
      {subtitle ? (
        <p className="mt-0.5 text-xs text-gray-500">{subtitle}</p>
      ) : null}
    </div>
  );
}

const bandStyles: Record<string, string> = {
  "Excellent match": "bg-green-50 text-green-700 border-green-200",
  "Good match": "bg-blue-50 text-blue-700 border-blue-200",
  "Moderate match": "bg-amber-50 text-amber-700 border-amber-200",
  "Fair match": "bg-orange-50 text-orange-700 border-orange-200",
  "Weak match": "bg-red-50 text-red-700 border-red-200",
};

export function BandBadge({ band }: { band: string | null | undefined }) {
  if (!band) return null;
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium",
        bandStyles[band] ?? "bg-gray-50 text-gray-600 border-gray-200"
      )}
    >
      {band}
    </span>
  );
}

/**
 * Score ring (0–100). SVG stroke-dasharray ring — no chart lib needed
 * for a single number, and it prints/renders identically everywhere.
 */
export function ScoreDial({
  percent,
  size = 120,
}: {
  percent: number;
  size?: number;
}) {
  const stroke = 10;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(100, percent));
  const offset = circumference * (1 - clamped / 100);
  const color =
    clamped >= 70 ? "#15803d" : clamped >= 55 ? "#1d4ed8" : clamped >= 40 ? "#b45309" : "#b91c1c";

  return (
    <div className="relative inline-flex" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-semibold text-gray-900">{clamped}</span>
        <span className="text-[10px] uppercase tracking-wide text-gray-400">
          / 100
        </span>
      </div>
    </div>
  );
}

/** Horizontal bar for a 0–1 component score. */
export function ScoreBar({
  label,
  value,
  weight,
  evidence,
}: {
  label: string;
  value: number; // 0..1
  weight?: number;
  evidence?: string;
}) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
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
        <span className="tabular-nums font-medium text-gray-900">{pct}</span>
      </div>
      <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-gray-100">
        <div
          className="h-full rounded-full bg-gray-700"
          style={{ width: `${pct}%` }}
        />
      </div>
      {evidence ? (
        <p className="mt-1 text-xs text-gray-500">{evidence}</p>
      ) : null}
    </div>
  );
}

export function Loading({ label = "Analyzing…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-sm text-gray-500">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-gray-700" />
      {label}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}
