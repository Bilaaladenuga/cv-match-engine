/**
 * MatchReportView — renders a full compatibility report.
 *
 * Used by /analyze (live POST /api/matches response) and
 * /history/[match_id] (stored GET /api/matches/{id} detail, mapped
 * into the same display shape by the page).
 *
 * Premium design: gradient score dial, animated bars, glassmorphic
 * sections, rich skill evidence table, and contextual charts.
 */

import {
  CheckCircle2,
  CircleDashed,
  MinusCircle,
  XCircle,
  TrendingUp,
  TrendingDown,
  Lightbulb,
  Shield,
  BarChart3,
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
  matched: <CheckCircle2 className="h-4 w-4 text-accent-500" />,
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
  strong: "text-accent-700 bg-accent-50",
  moderate: "text-amber-700 bg-amber-50",
  weak: "text-orange-700 bg-orange-50",
  absent: "text-gray-500 bg-gray-50",
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
    <tr className="border-b border-gray-100/60 transition-colors hover:bg-gray-50/50">
      <td className="py-3 pr-4 align-middle">
        <div className="flex items-center gap-2.5">
          {statusIcon[ev.status] ?? statusIcon.unknown}
          <span className="font-medium text-gray-900">{ev.skill}</span>
        </div>
      </td>
      <td className="py-3 pr-4">
        <span className="text-sm text-gray-600">{capitalize(ev.status)}</span>
      </td>
      <td className="py-3 pr-4">
        <span
          className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${strengthStyle[ev.strength] ?? "text-gray-500 bg-gray-50"}`}
        >
          {strengthLabel[ev.strength] ?? capitalize(ev.strength)}
        </span>
        {ev.strength === "weak" && ev.in_skills_section ? (
          <span className="mt-1 block text-[11px] text-orange-500">
            Listed in skills section, little work-history evidence
          </span>
        ) : null}
        {months ? (
          <span className="mt-0.5 block text-[11px] text-gray-400">
            ≈{months} of use
          </span>
        ) : null}
      </td>
      <td className="py-3 text-xs text-gray-400">
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
// Charts
// ---------------------------------------------------------------------------

const CHART_COLORS = {
  matched: "#059669",
  partial: "#D97706",
  missing: "#DC2626",
  unknown: "#9CA3AF",
} as const;

const COMPONENT_COLORS: Record<string, string> = {
  skills: "#2563EB",
  semantic: "#7C3AED",
  experience: "#059669",
  education: "#D97706",
  certifications: "#EC4899",
  ml_model: "#6366F1",
};

interface Factor {
  feature: string;
  label: string;
  contribution: number;
  direction: "helps" | "hurts" | "neutral";
  value: number;
  reference: number;
  detail: string;
}

function FeatureImportanceChart({ factors }: { factors: Factor[] }) {
  if (!factors || factors.length === 0) return null;

  // Sort by absolute contribution, take top 10
  const sorted = [...factors]
    .filter((f) => f.direction !== "neutral" && Math.abs(f.contribution) > 0.001)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 10);

  if (sorted.length === 0) return null;

  const data = sorted.map((f) => ({
    name: f.label.length > 25 ? f.label.slice(0, 22) + "..." : f.label,
    contribution: Math.round(f.contribution * 100),
    direction: f.direction,
    fill: f.direction === "helps" ? "#059669" : "#DC2626",
  }));

  return (
    <div>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: "#9CA3AF", fontFamily: "DM Sans" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `${v > 0 ? "+" : ""}${v}`}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={180}
              tick={{ fontSize: 11, fill: "#374151", fontFamily: "DM Sans" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: "rgba(37, 99, 235, 0.04)" }}
              contentStyle={{
                borderRadius: 12,
                border: "1px solid #E2E8F0",
                fontSize: 12,
                fontFamily: "DM Sans",
                boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
              }}
              formatter={(value: number) => [
                `${value > 0 ? "+" : ""}${value} pts`,
                "contribution",
              ]}
            />
            <Bar dataKey="contribution" radius={[0, 6, 6, 0]} barSize={16}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex items-center gap-4 text-xs text-gray-400">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-accent-500" />
          Helped your score
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
          Hurt your score
        </span>
      </div>
    </div>
  );
}

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
      fill: COMPONENT_COLORS[c.name] ?? "#6B7280",
    }))
    .sort((a, b) => b.contribution - a.contribution);

  return (
    <div>
      <div className="h-44 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
            <XAxis
              type="number"
              domain={[0, Math.max(50, overallPercent)]}
              tick={{ fontSize: 11, fill: "#9CA3AF", fontFamily: "DM Sans" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={100}
              tick={{ fontSize: 12, fill: "#374151", fontFamily: "Space Grotesk", fontWeight: 500 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: "rgba(37, 99, 235, 0.04)" }}
              contentStyle={{
                borderRadius: 12,
                border: "1px solid #E2E8F0",
                fontSize: 12,
                fontFamily: "DM Sans",
                boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
              }}
              formatter={(value: number) => [`${value} pts`, "contribution"]}
            />
            <Bar dataKey="contribution" radius={[0, 6, 6, 0]} barSize={20}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
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

function SkillCoverageDonut({ evidence }: { evidence: SkillEvidence[] }) {
  const counts = {
    matched: evidence.filter((e) => e.status === "matched").length,
    partial: evidence.filter((e) => e.status === "partial").length,
    missing: evidence.filter((e) => e.status === "missing").length,
    unknown: evidence.filter((e) => e.status === "unknown").length,
  };
  const data = (
    Object.entries(counts) as [
      keyof typeof CHART_COLORS,
      number,
    ][]
  )
    .filter(([, n]) => n > 0)
    .map(([status, n]) => ({
      name: status === "unknown" ? "Unknown" : capitalize(status),
      value: n,
      fill: CHART_COLORS[status],
    }));
  const total = evidence.length;

  return (
    <div className="flex items-center gap-6 px-6 py-5">
      <div className="relative h-40 w-40 shrink-0">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              innerRadius={48}
              outerRadius={70}
              paddingAngle={3}
              strokeWidth={0}
            >
              {data.map((d) => (
                <Cell key={d.name} fill={d.fill} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                borderRadius: 12,
                border: "1px solid #E2E8F0",
                fontSize: 12,
                fontFamily: "DM Sans",
                boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
              }}
              formatter={(value: number, name: string) => [
                `${value} of ${total} requirements`,
                name,
              ]}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-heading text-2xl font-bold text-gray-900">
            {total}
          </span>
          <span className="text-[10px] font-medium uppercase tracking-widest text-gray-400">
            required
          </span>
        </div>
      </div>
      <ul className="space-y-2">
        {(
          [
            ["matched", counts.matched],
            ["partial", counts.partial],
            ["missing", counts.missing],
            ["unknown", counts.unknown],
          ] as [keyof typeof CHART_COLORS, number][]
        )
          .filter(([, n]) => n > 0)
          .map(([status, n]) => (
            <li
              key={status}
              className="flex items-center gap-2.5 text-sm text-gray-700"
            >
              <span
                className="h-3 w-3 rounded-sm"
                style={{ backgroundColor: CHART_COLORS[status] }}
              />
              {capitalize(status)}
              <span className="font-semibold tabular-nums">{n}</span>
              <span className="text-xs text-gray-400">
                ({Math.round((n / total) * 100)}%)
              </span>
            </li>
          ))}
      </ul>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

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
          <h2 className="font-heading text-xl font-bold tracking-tight text-gray-900">
            {title}
          </h2>
          {subtitle ? (
            <p className="mt-0.5 text-sm text-gray-500">{subtitle}</p>
          ) : null}
        </div>
      ) : null}

      {/* Overall Score */}
      <Card className="overflow-hidden">
        <div className="flex flex-col items-center gap-8 px-8 py-8 sm:flex-row">
          <ScoreDial percent={report.overall_percent} />
          <div className="flex-1 text-center sm:text-left">
            <div className="flex items-center justify-center gap-3 sm:justify-start">
              <BandBadge band={report.band} />
            </div>
            <p className="mt-3 text-sm leading-relaxed text-gray-600">
              {report.overall_percent >= 70
                ? "Strong overlap between the CV and this job's requirements."
                : report.overall_percent >= 55
                  ? "Reasonable overlap with some gaps to close."
                  : "Significant gaps between the CV and this job's requirements."}
            </p>
          </div>
        </div>
      </Card>

      {/* Score Breakdown */}
      <Card>
        <CardHeader
          title="Score breakdown"
          subtitle="Each component's raw score and its weight in the overall score"
        />
        <div className="space-y-5 px-6 py-5">
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
        <div className="border-t border-gray-100/60 px-4 py-5">
          <ScoreContributionChart
            components={report.components}
            overallPercent={report.overall_percent}
          />
        </div>
      </Card>

      {/* Skill Evidence */}
      <Card>
        <CardHeader
          title="Required skills"
          subtitle="Match status and the evidence behind it, per job requirement"
        />
        <div className="border-b border-gray-100/60">
          <SkillCoverageDonut evidence={evidence} />
        </div>
        <div className="overflow-x-auto px-5 py-2">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-gray-200/60 text-[11px] font-semibold uppercase tracking-widest text-gray-400">
                <th className="py-2.5 pr-4 font-semibold">Skill</th>
                <th className="py-2.5 pr-4 font-semibold">Status</th>
                <th className="py-2.5 pr-4 font-semibold">Evidence</th>
                <th className="py-2.5 font-semibold">Sources</th>
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

      {/* Positive / Negative Factors */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Positive factors" />
          <ul className="space-y-2.5 px-5 py-4">
            {report.positive_factors.length === 0 ? (
              <li className="text-sm text-gray-400">None recorded.</li>
            ) : (
              report.positive_factors.map((f, i) => (
                <li
                  key={i}
                  className="flex items-start gap-3 text-sm text-gray-700"
                >
                  <TrendingUp className="mt-0.5 h-4 w-4 shrink-0 text-accent-500" />
                  {f}
                </li>
              ))
            )}
          </ul>
        </Card>
        <Card>
          <CardHeader title="Negative factors" />
          <ul className="space-y-2.5 px-5 py-4">
            {report.negative_factors.length === 0 ? (
              <li className="text-sm text-gray-400">None recorded.</li>
            ) : (
              report.negative_factors.map((f, i) => (
                <li
                  key={i}
                  className="flex items-start gap-3 text-sm text-gray-700"
                >
                  <TrendingDown className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
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
          title="How to improve your CV"
          subtitle="Actionable steps to better match this job — based on what the system found"
        />
        <ul className="space-y-3 px-5 py-4">
          {report.recommendations.length === 0 ? (
            <li className="text-sm text-gray-400">No actions needed.</li>
          ) : (
            report.recommendations.map((r, i) => (
              <li
                key={i}
                className="flex items-start gap-3 text-sm text-gray-700"
              >
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary-600/10 text-[10px] font-bold text-primary-700">
                  {i + 1}
                </span>
                {r}
              </li>
            ))
          )}
        </ul>
      </Card>

      {/* AI Fit Prediction — plain language for non-technical users */}
      {ml && ml.label !== undefined ? (
        <Card className="overflow-hidden">
          <div className="bg-gradient-to-r from-primary-700 to-primary-600 px-6 py-4">
            <h3 className="font-heading text-sm font-semibold text-white">
              AI Fit Prediction
            </h3>
            <p className="mt-0.5 text-xs text-white/70">
              A trained model that learned from thousands of CV–Job pairs
            </p>
          </div>
          <div className="px-6 py-5">
            {/* Main prediction in plain language */}
            <div className="mb-5 flex items-center gap-4 rounded-xl bg-gray-50/80 px-5 py-4">
              <div
                className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl text-lg font-bold text-white ${
                  ml.label === "Good Fit"
                    ? "bg-gradient-to-br from-accent-500 to-emerald-600"
                    : ml.label === "Potential Fit"
                      ? "bg-gradient-to-br from-amber-500 to-orange-500"
                      : "bg-gradient-to-br from-red-500 to-red-600"
                }`}
              >
                {ml.label === "Good Fit"
                  ? "✓"
                  : ml.label === "Potential Fit"
                    ? "~"
                    : "✗"}
              </div>
              <div>
                <p className="font-heading text-base font-semibold text-gray-900">
                  {ml.label === "Good Fit"
                    ? "This CV is a strong match for this job"
                    : ml.label === "Potential Fit"
                      ? "This CV partially matches — some gaps to address"
                      : "This CV has significant gaps for this job"}
                </p>
                <p className="mt-0.5 text-sm text-gray-500">
                  {ml.label === "Good Fit"
                    ? "The model is confident this candidate would be a good fit based on skills, experience, and education."
                    : ml.label === "Potential Fit"
                      ? "The model sees some matching elements but also gaps that could be addressed."
                      : "The model predicts this candidate would need significant development to meet this role's requirements."}
                </p>
              </div>
            </div>

            {/* Confidence breakdown — plain language */}
            {ml.probabilities ? (
              <div>
                <p className="mb-3 text-xs font-medium uppercase tracking-wider text-gray-400">
                  Confidence breakdown
                </p>
                <div className="space-y-3">
                  {Object.entries(ml.probabilities).map(([cls, p]) => {
                    const label =
                      cls === "Good Fit"
                        ? "Strong match"
                        : cls === "Potential Fit"
                          ? "Partial match"
                          : "Not a match";
                    const desc =
                      cls === "Good Fit"
                        ? "How likely this CV is a strong fit"
                        : cls === "Potential Fit"
                          ? "How likely this CV has potential with some gaps"
                          : "How likely this CV doesn't match well";
                    return (
                      <div key={cls}>
                        <div className="flex items-baseline justify-between text-sm">
                          <span className="font-medium text-gray-700">
                            {label}
                          </span>
                          <span className="tabular-nums font-semibold text-gray-900">
                            {Math.round(p * 100)}%
                          </span>
                        </div>
                        <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-gray-100">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ease-out ${
                              cls === "Good Fit"
                                ? "bg-gradient-to-r from-accent-500 to-emerald-500"
                                : cls === "Potential Fit"
                                  ? "bg-gradient-to-r from-amber-500 to-orange-400"
                                  : "bg-gradient-to-r from-red-400 to-red-500"
                            }`}
                            style={{ width: `${Math.round(p * 100)}%` }}
                          />
                        </div>
                        <p className="mt-0.5 text-[11px] text-gray-400">{desc}</p>
                      </div>
                    );
                  })}
                </div>
                {ml.raw_probabilities &&
                  JSON.stringify(ml.raw_probabilities) !==
                    JSON.stringify(ml.probabilities) && (
                    <p className="mt-3 text-[11px] text-gray-400">
                      These predictions are adjusted for real-world job market
                      distributions.
                    </p>
                  )}
              </div>
            ) : (
              <p className="text-sm text-gray-500">
                The AI model was not available for this analysis. The score
                above comes entirely from the rule-based matching engine.
              </p>
            )}
          </div>
        </Card>
      ) : null}

      {/* Feature Importance — what helped and what hurt */}
      {ml?.explanation?.factors && ml.explanation.factors.length > 0 ? (
        <Card>
          <CardHeader
            title="What affected your score"
            subtitle="The top factors that helped or hurt your match — based on how your CV compares to typical candidates"
          />
          <div className="px-6 py-5">
            <FeatureImportanceChart factors={ml.explanation.factors as Factor[]} />
          </div>
          {/* Detail list for mobile / accessibility */}
          <div className="border-t border-gray-100/60 px-5 py-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-gray-400">
              Detailed breakdown
            </p>
            <ul className="space-y-2">
              {(ml.explanation.factors as Factor[])
                .filter((f) => f.direction !== "neutral" && Math.abs(f.contribution) > 0.001)
                .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                .slice(0, 8)
                .map((f, i) => (
                  <li
                    key={`${f.feature}-${i}`}
                    className="flex items-start gap-2.5 text-sm text-gray-600"
                  >
                    {f.direction === "helps" ? (
                      <TrendingUp className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent-500" />
                    ) : (
                      <TrendingDown className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-400" />
                    )}
                    <span>
                      <span className="font-medium text-gray-900">{f.label}</span>
                      <span className="ml-1 text-gray-500">
                        ({f.contribution > 0 ? "+" : ""}
                        {Math.round(f.contribution * 100)} pts)
                      </span>
                    </span>
                  </li>
                ))}
            </ul>
          </div>
        </Card>
      ) : null}

      {/* Ethics Disclaimer */}
      <div className="flex items-start gap-3 rounded-xl border border-gray-200/60 bg-gray-50/80 px-5 py-4">
        <Shield className="mt-0.5 h-4 w-4 shrink-0 text-gray-400" />
        <p className="text-xs leading-relaxed text-gray-500">
          {report.disclaimer}
        </p>
      </div>
    </div>
  );
}
