/**
 * MatchReportView — renders a full compatibility report.
 *
 * Used by /analyze (live POST /api/matches response) and
 * /history/[match_id] (stored GET /api/matches/{id} detail, mapped
 * into the same display shape by the page).
 *
 * Design notes:
 * - Every score is shown with its evidence; the ethics disclaimer is
 *   always visible (Phase 27 requirement, not an afterthought).
 * - The skill table surfaces the Phase 16 evidence grades directly:
 *   status (matched/partial/missing) × strength (strong/moderate/weak),
 *   so "listed but not evidenced" padding is visible at a glance.
 * - ML probabilities are shown with their calibration method, raw and
 *   corrected, per the Phase 14 explainability contract.
 */

import {
  CheckCircle2,
  CircleDashed,
  MinusCircle,
  XCircle,
} from "lucide-react";
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MatchReport, SkillEvidence } from "@/lib/types";
import {
  BandBadge,
  Card,
  CardHeader,
  ScoreBar,
  ScoreDial,
} from "./ui";

const statusIcon = {
  matched: <CheckCircle2 className="h-4 w-4 text-green-600" />,
  partial: <CircleDashed className="h-4 w-4 text-amber-500" />,
  missing: <XCircle className="h-4 w-4 text-red-500" />,
  unknown: <MinusCircle className="h-4 w-4 text-gray-400" />,
} as const;

const strengthLabel: Record<string, string> = {
  strong: "Strong evidence",
  moderate: "Moderate evidence",
  weak: "Weak evidence",
  absent: "No evidence",
};

const strengthStyle: Record<string, string> = {
  strong: "text-green-700",
  moderate: "text-amber-600",
  weak: "text-orange-600",
  absent: "text-gray-400",
};

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function SkillEvidenceRow({ ev }: { ev: SkillEvidence }) {
  const months =
    ev.estimated_months !== null && ev.estimated_months !== undefined
      ? `${ev.estimated_months} mo${ev.estimated_months === 1 ? "" : "s"}`
      : null;
  return (
    <tr className="border-b border-gray-50 last:border-0">
      <td className="py-2 pr-3 align-middle">
        <div className="flex items-center gap-2">
          {statusIcon[ev.status] ?? statusIcon.unknown}
          <span className="font-medium text-gray-900">{ev.skill}</span>
        </div>
      </td>
      <td className="py-2 pr-3 text-sm text-gray-600">
        {capitalize(ev.status)}
      </td>
      <td className={`py-2 pr-3 text-sm ${strengthStyle[ev.strength] ?? "text-gray-500"}`}>
        {strengthLabel[ev.strength] ?? capitalize(ev.strength)}
        {ev.strength === "weak" && ev.in_skills_section ? (
          <span className="block text-xs text-orange-500">
            Listed in skills section, little work-history evidence
          </span>
        ) : null}
        {months ? (
          <span className="block text-xs text-gray-400">≈{months} of use</span>
        ) : null}
      </td>
      <td className="py-2 text-xs text-gray-400">
        {ev.sources && ev.sources.length > 0 ? ev.sources.join("; ") : "—"}
      </td>
    </tr>
  );
}

function fallbackEvidence(
  matched: string[],
  partial: string[],
  missing: string[]
): SkillEvidence[] {
  // Stored reports (pre-Phase 16) only have the three skill lists.
  return [
    ...matched.map((s) => ({
      skill: s,
      status: "matched" as const,
      candidate_skill: s,
      strength: "strong" as const,
      in_skills_section: true,
      in_work_history: true,
      estimated_months: null,
      sources: [],
    })),
    ...partial.map((s) => ({
      skill: s,
      status: "partial" as const,
      candidate_skill: s,
      strength: "moderate" as const,
      in_skills_section: true,
      in_work_history: false,
      estimated_months: null,
      sources: [],
    })),
    ...missing.map((s) => ({
      skill: s,
      status: "missing" as const,
      candidate_skill: null,
      strength: "absent" as const,
      in_skills_section: false,
      in_work_history: false,
      estimated_months: null,
      sources: [],
    })),
  ];
}

export interface ReportLists {
  matched: string[];
  partial: string[];
  missing: string[];
}

// ---------------------------------------------------------------------------
// Charts (Phase 18) — every chart carries information the text lists don't.
// ---------------------------------------------------------------------------

const COVERAGE_COLORS = {
  matched: "#16a34a", // green-600
  partial: "#d97706", // amber-600
  missing: "#dc2626", // red-500
  unknown: "#9ca3af", // gray-400
} as const;

/**
 * Weighted-contribution chart: the component bars elsewhere show RAW
 * scores; this shows where the overall score's points actually came from
 * (raw × weight, sums to ~overall%). Answers "what carried my score".
 */
function ScoreContributionChart({
  components,
  overallPercent,
}: {
  components: MatchReport["components"];
  overallPercent: number;
}) {
  const data = components
    .map((c) => ({
      name: capitalize(c.name),
      contribution: Math.round(c.weighted * 100),
      detail: `raw ${Math.round(c.raw_score * 100)}/100 × weight ${Math.round(
        c.weight * 100
      )}%`,
    }))
    .sort((a, b) => b.contribution - a.contribution);

  return (
    <div>
      <div className="h-40 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
            <XAxis
              type="number"
              domain={[0, Math.max(50, overallPercent)]}
              tick={{ fontSize: 11, fill: "#9ca3af" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={92}
              tick={{ fontSize: 12, fill: "#374151" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: "#f9fafb" }}
              formatter={(value: number) => [`${value} pts`, "contribution"]}
              labelStyle={{ color: "#111827", fontWeight: 500 }}
              contentStyle={{
                borderRadius: 6,
                border: "1px solid #e5e7eb",
                fontSize: 12,
              }}
            />
            <Bar dataKey="contribution" fill="#374151" radius={[0, 3, 3, 0]} barSize={18} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-xs text-gray-400">
        Points contributed to the overall score (sums to ≈{overallPercent}
        /100). Hover a bar for its raw score × weight.
      </p>
    </div>
  );
}

/**
 * Coverage donut: composition of job requirements by match status with
 * counts. Complements the table by making the size of each group visible
 * at a glance (12 matched / 2 partial / 1 missing reads instantly).
 */
function SkillCoverageDonut({ evidence }: { evidence: SkillEvidence[] }) {
  const counts = {
    matched: evidence.filter((e) => e.status === "matched").length,
    partial: evidence.filter((e) => e.status === "partial").length,
    missing: evidence.filter((e) => e.status === "missing").length,
    unknown: evidence.filter((e) => e.status === "unknown").length,
  };
  const data = (
    Object.entries(counts) as [
      keyof typeof COVERAGE_COLORS,
      number,
    ][]
  )
    .filter(([, n]) => n > 0)
    .map(([status, n]) => ({
      name: status === "unknown" ? "Unknown" : capitalize(status),
      value: n,
      color: COVERAGE_COLORS[status],
    }));
  const total = evidence.length;

  return (
    <div className="flex items-center gap-5 px-5 py-4">
      <div className="relative h-36 w-36 shrink-0">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              innerRadius={44}
              outerRadius={66}
              paddingAngle={2}
              strokeWidth={0}
            >
              {data.map((d) => (
                <Cell key={d.name} fill={d.color} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value: number, name: string) => [
                `${value} of ${total} requirements`,
                name,
              ]}
              contentStyle={{
                borderRadius: 6,
                border: "1px solid #e5e7eb",
                fontSize: 12,
              }}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-semibold text-gray-900">{total}</span>
          <span className="text-[10px] uppercase tracking-wide text-gray-400">
            required
          </span>
        </div>
      </div>
      <ul className="space-y-1.5 text-sm">
        {(
          [
            ["matched", counts.matched],
            ["partial", counts.partial],
            ["missing", counts.missing],
            ["unknown", counts.unknown],
          ] as [keyof typeof COVERAGE_COLORS, number][]
        )
          .filter(([, n]) => n > 0)
          .map(([status, n]) => (
            <li key={status} className="flex items-center gap-2 text-gray-700">
              <span
                className="h-2.5 w-2.5 rounded-sm"
                style={{ backgroundColor: COVERAGE_COLORS[status] }}
              />
              {capitalize(status)}
              <span className="font-medium tabular-nums">{n}</span>
              <span className="text-xs text-gray-400">
                ({Math.round((n / total) * 100)}%)
              </span>
            </li>
          ))}
      </ul>
    </div>
  );
}

export function MatchReportView({
  report,
  lists,
  title,
  subtitle,
}: {
  report: Pick<
    MatchReport,
    | "model_version"
    | "overall_score"
    | "overall_percent"
    | "band"
    | "components"
    | "weights"
    | "positive_factors"
    | "negative_factors"
    | "recommendations"
    | "disclaimer"
    | "skill_evidence"
    | "ml_details"
  >;
  lists: ReportLists;
  title?: string;
  subtitle?: string;
}) {
  const evidence =
    report.skill_evidence && report.skill_evidence.length > 0
      ? report.skill_evidence
      : fallbackEvidence(lists.matched, lists.partial, lists.missing);

  const ml = report.ml_details;

  return (
    <div className="space-y-6">
      {title ? (
        <div>
          <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
          {subtitle ? (
            <p className="text-sm text-gray-500">{subtitle}</p>
          ) : null}
        </div>
      ) : null}

      {/* Overall */}
      <Card>
        <div className="flex flex-col items-center gap-6 px-6 py-6 sm:flex-row">
          <ScoreDial percent={report.overall_percent} />
          <div className="flex-1 text-center sm:text-left">
            <div className="flex items-center justify-center gap-2 sm:justify-start">
              <BandBadge band={report.band} />
              <span className="text-xs text-gray-400">
                model {report.model_version}
              </span>
            </div>
            <p className="mt-2 text-sm text-gray-600">
              {report.overall_percent >= 70
                ? "Strong overlap between the CV and this job's requirements."
                : report.overall_percent >= 55
                  ? "Reasonable overlap with some gaps to close."
                  : "Significant gaps between the CV and this job's requirements."}
            </p>
          </div>
        </div>
      </Card>

      {/* Component breakdown */}
      <Card>
        <CardHeader
          title="Score breakdown"
          subtitle="Each component's raw score and its weight in the overall score"
        />
        <div className="space-y-4 px-5 py-4">
          {report.components.map((c) => (
            <ScoreBar
              key={c.name}
              label={capitalize(c.name)}
              value={c.raw_score}
              weight={c.weight}
              evidence={c.evidence}
            />
          ))}
        </div>
        <div className="border-t border-gray-100 px-2 py-4">
          <ScoreContributionChart
            components={report.components}
            overallPercent={report.overall_percent}
          />
        </div>
      </Card>

      {/* Skill evidence table */}
      <Card>
        <CardHeader
          title="Required skills"
          subtitle="Match status and the evidence behind it, per job requirement"
        />
        <div className="border-b border-gray-100">
          <SkillCoverageDonut evidence={evidence} />
        </div>
        <div className="overflow-x-auto px-5 py-2">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-gray-200 text-xs uppercase tracking-wide text-gray-400">
                <th className="py-2 pr-3 font-medium">Skill</th>
                <th className="py-2 pr-3 font-medium">Status</th>
                <th className="py-2 pr-3 font-medium">Evidence</th>
                <th className="py-2 font-medium">Sources</th>
              </tr>
            </thead>
            <tbody>
              {evidence.map((ev, i) => (
                <SkillEvidenceRow key={`${ev.skill}-${i}`} ev={ev} />
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Factors */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Positive factors" />
          <ul className="space-y-2 px-5 py-4">
            {report.positive_factors.length === 0 ? (
              <li className="text-sm text-gray-400">None recorded.</li>
            ) : (
              report.positive_factors.map((f, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                  <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-green-500" />
                  {f}
                </li>
              ))
            )}
          </ul>
        </Card>
        <Card>
          <CardHeader title="Negative factors" />
          <ul className="space-y-2 px-5 py-4">
            {report.negative_factors.length === 0 ? (
              <li className="text-sm text-gray-400">None recorded.</li>
            ) : (
              report.negative_factors.map((f, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                  <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-red-400" />
                  {f}
                </li>
              ))
            )}
          </ul>
        </Card>
      </div>

      {/* Recommendations */}
      <Card>
        <CardHeader
          title="Recommendations"
          subtitle="Grounded in what the parser actually found on the CV"
        />
        <ul className="space-y-2 px-5 py-4">
          {report.recommendations.length === 0 ? (
            <li className="text-sm text-gray-400">No actions needed.</li>
          ) : (
            report.recommendations.map((r, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
                {r}
              </li>
            ))
          )}
        </ul>
      </Card>

      {/* ML model internals (transparency, Phase 14/23) */}
      {ml && ml.label !== undefined ? (
        <Card>
          <CardHeader
            title="Trained model view"
            subtitle={`Classifier: ${ml.label} · calibration: ${
              ml.calibration_method ?? "none"
            }`}
          />
          <div className="px-5 py-4">
            {ml.probabilities ? (
              <div className="space-y-3">
                {Object.entries(ml.probabilities).map(([cls, p]) => (
                  <ScoreBar
                    key={cls}
                    label={`P(${cls})`}
                    value={p}
                  />
                ))}
                {ml.raw_probabilities &&
                  JSON.stringify(ml.raw_probabilities) !==
                    JSON.stringify(ml.probabilities) && (
                    <p className="text-xs text-gray-400">
                      Probabilities prior-corrected for the natural class
                      distribution; raw values available via the API.
                    </p>
                  )}
              </div>
            ) : (
              <p className="text-sm text-gray-500">
                The trained model was not available for this analysis; scores
                come from the deterministic hybrid engine.
              </p>
            )}
          </div>
        </Card>
      ) : null}

      {/* Ethics disclaimer — always visible */}
      <p className="rounded-md border border-gray-200 bg-gray-50 px-4 py-3 text-xs leading-relaxed text-gray-500">
        {report.disclaimer}
      </p>
    </div>
  );
}
