import {
  CheckCircle2,
  CircleDashed,
  MinusCircle,
  XCircle,
  TrendingUp,
  TrendingDown,
  Shield,
  Lightbulb,
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
import { BandBadge, Card, CardHeader } from "./ui";

const statusIcon = {
  matched: <CheckCircle2 className="h-4 w-4 text-carbon" />,
  partial: <CircleDashed className="h-4 w-4 text-smoke" />,
  missing: <XCircle className="h-4 w-4 text-carbon" />,
  unknown: <MinusCircle className="h-4 w-4 text-ash" />,
} as const;

const strengthLabel: Record<string, string> = {
  strong: "Strong",
  moderate: "Moderate",
  weak: "Weak",
  absent: "None",
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
    <tr className="border-b border-ash/30 transition-colors hover:bg-mist">
      <td className="py-3 pr-4 align-middle">
        <div className="flex items-center gap-2.5">
          {statusIcon[ev.status] ?? statusIcon.unknown}
          <span className="font-medium text-carbon">{ev.skill}</span>
        </div>
      </td>
      <td className="py-3 pr-4">
        <span className="text-body-sm text-slate">{capitalize(ev.status)}</span>
      </td>
      <td className="py-3 pr-4">
        <span className="inline-flex items-center rounded-tag bg-mint/30 px-3 py-0.5 text-caption font-mono text-carbon">
          {strengthLabel[ev.strength] ?? capitalize(ev.strength)}
        </span>
        {months ? (
          <span className="mt-0.5 block text-caption text-smoke">{months}</span>
        ) : null}
      </td>
      <td className="py-3 text-caption text-smoke">
        {ev.sources && ev.sources.length > 0 ? ev.sources.join("; ") : "-"}
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

const CHART_COLORS = {
  matched: "#000000",
  partial: "#979797",
  missing: "#c6c6c6",
  unknown: "#e5e5e5",
} as const;

const COMPONENT_COLORS: Record<string, string> = {
  skills: "#000000",
  semantic: "#444444",
  experience: "#2f2f2f",
  education: "#979797",
  certifications: "#c6c6c6",
  ml_model: "#000000",
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

  const sorted = [...factors]
    .filter((f) => f.direction !== "neutral" && Math.abs(f.contribution) > 0.001)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 10);

  if (sorted.length === 0) return null;

  const data = sorted.map((f) => ({
    name: f.label.length > 25 ? f.label.slice(0, 22) + "..." : f.label,
    contribution: Math.round(f.contribution * 100),
    direction: f.direction,
    fill: f.direction === "helps" ? "#000000" : "#c6c6c6",
  }));

  return (
    <div>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: "#979797", fontFamily: "Inter" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `${v > 0 ? "+" : ""}${v}`}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={180}
              tick={{ fontSize: 11, fill: "#444444", fontFamily: "Inter" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: "rgba(0, 0, 0, 0.04)" }}
              contentStyle={{
                borderRadius: 12,
                border: "1px solid #c6c6c6",
                fontSize: 12,
                fontFamily: "Inter",
              }}
              formatter={(value: number) => [
                `${value > 0 ? "+" : ""}${value} pts`,
                "contribution",
              ]}
            />
            <Bar dataKey="contribution" radius={[0, 4, 4, 0]} barSize={16}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex items-center gap-4 text-caption text-smoke">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-carbon" />
          Helped your score
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-ash" />
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
      fill: COMPONENT_COLORS[c.name] ?? "#979797",
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
              tick={{ fontSize: 11, fill: "#979797", fontFamily: "Inter" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={100}
              tick={{ fontSize: 12, fill: "#444444", fontFamily: "Inter", fontWeight: 500 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: "rgba(0, 0, 0, 0.04)" }}
              contentStyle={{
                borderRadius: 12,
                border: "1px solid #c6c6c6",
                fontSize: 12,
                fontFamily: "Inter",
              }}
              formatter={(value: number) => [`${value} pts`, "contribution"]}
            />
            <Bar dataKey="contribution" radius={[0, 4, 4, 0]} barSize={20}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 label-mono text-smoke">
        Points contributed to the overall score. Hover a bar for details.
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
    Object.entries(counts) as [keyof typeof CHART_COLORS, number][]
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
                border: "1px solid #c6c6c6",
                fontSize: 12,
                fontFamily: "Inter",
              }}
              formatter={(value: number, name: string) => [
                `${value} of ${total}`,
                name,
              ]}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-display text-heading text-carbon">{total}</span>
          <span className="label-mono text-smoke">required</span>
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
              className="flex items-center gap-2.5 text-body text-slate"
            >
              <span
                className="h-3 w-3 rounded-sm"
                style={{ backgroundColor: CHART_COLORS[status] }}
              />
              {capitalize(status)}
              <span className="font-medium tabular-nums">{n}</span>
              <span className="label-mono text-smoke">
                ({Math.round((n / total) * 100)}%)
              </span>
            </li>
          ))}
      </ul>
    </div>
  );
}

function ScoreDial({ percent }: { percent: number }) {
  const size = 120;
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percent / 100) * circumference;

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
        <span className="font-display text-heading text-carbon">{percent}</span>
        <span className="label-mono text-smoke">/100</span>
      </div>
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
          <h2 className="heading-display text-heading text-carbon">{title}</h2>
          {subtitle ? (
            <p className="mt-1 label-mono text-smoke">{subtitle}</p>
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
            <p className="mt-3 text-body text-slate">
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
          subtitle="Each component's raw score and its weight"
        />
        <div className="space-y-5 px-6 py-5">
          {report.components.map((c) => (
            <div key={c.name}>
              <div className="flex items-center justify-between mb-1">
                <span className="label-mono text-smoke">
                  {capitalize(c.name)} (weight: {Math.round(c.weight * 100)}%)
                </span>
                <span className="font-mono text-caption text-carbon">
                  {Math.round(c.raw_score * 100)}%
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-ash/30 overflow-hidden">
                <div
                  className="h-full rounded-full bg-carbon transition-all"
                  style={{ width: `${Math.round(c.raw_score * 100)}%` }}
                />
              </div>
              {c.evidence ? (
                <p className="mt-1 label-mono text-smoke">{c.evidence}</p>
              ) : null}
            </div>
          ))}
        </div>
        <div className="border-t border-ash/30 px-4 py-5">
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
          subtitle="Match status and evidence for each job requirement"
        />
        <div className="border-b border-ash/30">
          <SkillCoverageDonut evidence={evidence} />
        </div>
        <div className="overflow-x-auto px-5 py-2">
          <table className="w-full text-left text-body-sm">
            <thead>
              <tr className="border-b border-ash/30 label-mono text-smoke">
                <th className="py-2.5 pr-4">Skill</th>
                <th className="py-2.5 pr-4">Status</th>
                <th className="py-2.5 pr-4">Evidence</th>
                <th className="py-2.5">Sources</th>
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
              <li className="text-body-sm text-smoke">None recorded.</li>
            ) : (
              report.positive_factors.map((f, i) => (
                <li
                  key={i}
                  className="flex items-start gap-3 text-body-sm text-slate"
                >
                  <TrendingUp className="mt-0.5 h-4 w-4 shrink-0 text-carbon" />
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
              <li className="text-body-sm text-smoke">None recorded.</li>
            ) : (
              report.negative_factors.map((f, i) => (
                <li
                  key={i}
                  className="flex items-start gap-3 text-body-sm text-slate"
                >
                  <TrendingDown className="mt-0.5 h-4 w-4 shrink-0 text-smoke" />
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
          subtitle="Steps to better match this job, based on what the system found"
        />
        <ul className="space-y-3 px-5 py-4">
          {report.recommendations.length === 0 ? (
            <li className="text-body-sm text-smoke">No actions needed.</li>
          ) : (
            report.recommendations.map((r, i) => (
              <li
                key={i}
                className="flex items-start gap-3 text-body-sm text-slate"
              >
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-carbon text-paper text-caption font-mono">
                  {i + 1}
                </span>
                {r}
              </li>
            ))
          )}
        </ul>
      </Card>

      {/* AI Fit Prediction */}
      {ml && ml.label !== undefined ? (
        <Card className="overflow-hidden">
          <div className="bg-carbon px-6 py-4">
            <h3 className="font-body text-sub font-medium uppercase text-paper">
              AI Fit Prediction
            </h3>
            <p className="mt-1 label-mono text-smoke">
              Trained on thousands of CV-job pairs
            </p>
          </div>
          <div className="px-6 py-5">
            <div className="mb-5 flex items-center gap-4 rounded-card bg-mist px-5 py-4">
              <div
                className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-card text-lg font-bold text-paper ${
                  ml.label === "Good Fit"
                    ? "bg-carbon"
                    : ml.label === "Potential Fit"
                      ? "bg-graphite"
                      : "bg-ash"
                }`}
              >
                {ml.label === "Good Fit"
                  ? "✓"
                  : ml.label === "Potential Fit"
                    ? "~"
                    : "✗"}
              </div>
              <div>
                <p className="font-body text-sub font-medium text-carbon">
                  {ml.label === "Good Fit"
                    ? "This CV is a strong match for this job"
                    : ml.label === "Potential Fit"
                      ? "This CV partially matches, some gaps to address"
                      : "This CV has significant gaps for this job"}
                </p>
                <p className="mt-1 text-body-sm text-slate">
                  {ml.label === "Good Fit"
                    ? "The model is confident this candidate would be a good fit."
                    : ml.label === "Potential Fit"
                      ? "The model sees some matching elements but also gaps."
                      : "The model predicts this candidate needs significant development."}
                </p>
              </div>
            </div>

            {ml.probabilities ? (
              <div>
                <div className="mb-4 flex items-center justify-between">
                  <p className="label-mono text-smoke">Confidence breakdown</p>
                  {ml.confidence !== undefined && ml.confidence_note && (
                    <span className="tag">
                      {ml.confidence_note} confidence
                    </span>
                  )}
                </div>
                <div className="space-y-3">
                  {Object.entries(ml.probabilities).map(([cls, p]) => {
                    const label =
                      cls === "Good Fit"
                        ? "Strong match"
                        : cls === "Potential Fit"
                          ? "Partial match"
                          : "Not a match";
                    return (
                      <div key={cls}>
                        <div className="flex items-baseline justify-between text-body-sm">
                          <span className="font-medium text-carbon">{label}</span>
                          <span className="tabular-nums font-mono text-carbon">
                            {Math.round(p * 100)}%
                          </span>
                        </div>
                        <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-ash/30">
                          <div
                            className="h-full rounded-full bg-carbon transition-all duration-700"
                            style={{ width: `${Math.round(p * 100)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <p className="text-body-sm text-slate">
                The AI model was not available. The score comes from the rule-based engine.
              </p>
            )}
          </div>
        </Card>
      ) : null}

      {/* Feature Importance */}
      {ml?.explanation?.factors && ml.explanation.factors.length > 0 ? (
        <Card>
          <CardHeader
            title="What affected your score"
            subtitle="Top factors that helped or hurt your match"
          />
          <div className="px-6 py-5">
            <FeatureImportanceChart factors={ml.explanation.factors as Factor[]} />
          </div>
          <div className="border-t border-ash/30 px-5 py-4">
            <p className="mb-3 label-mono text-smoke">Detailed breakdown</p>
            <ul className="space-y-2">
              {(ml.explanation.factors as Factor[])
                .filter((f) => f.direction !== "neutral" && Math.abs(f.contribution) > 0.001)
                .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                .slice(0, 8)
                .map((f, i) => (
                  <li
                    key={`${f.feature}-${i}`}
                    className="flex items-start gap-2.5 text-body-sm text-slate"
                  >
                    {f.direction === "helps" ? (
                      <TrendingUp className="mt-0.5 h-3.5 w-3.5 shrink-0 text-carbon" />
                    ) : (
                      <TrendingDown className="mt-0.5 h-3.5 w-3.5 shrink-0 text-smoke" />
                    )}
                    <span>
                      <span className="font-medium text-carbon">{f.label}</span>
                      <span className="ml-1 text-smoke">
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

      {/* Disclaimer */}
      <div className="flex items-start gap-3 rounded-card bg-mist px-5 py-4">
        <Shield className="mt-0.5 h-4 w-4 shrink-0 text-smoke" />
        <p className="text-caption leading-relaxed text-smoke">
          {report.disclaimer}
        </p>
      </div>
    </div>
  );
}
