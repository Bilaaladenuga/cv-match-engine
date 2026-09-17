import Link from "next/link";
import {
  FileText,
  Brain,
  BarChart3,
  ArrowRight,
  CheckCircle2,
  Shield,
  Zap,
} from "lucide-react";

export default function HomePage() {
  return (
    <main className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden bg-gradient-hero">
        {/* Dot overlay */}
        <div className="absolute inset-0 dot-pattern-bg opacity-40" />
        {/* Gradient orbs */}
        <div className="absolute -left-32 -top-32 h-96 w-96 rounded-full bg-primary-600/20 blur-3xl" />
        <div className="absolute -bottom-32 -right-32 h-96 w-96 rounded-full bg-accent-500/15 blur-3xl" />

        <div className="relative mx-auto max-w-5xl px-6 py-24 text-center">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-xs font-medium text-white/70 backdrop-blur-sm">
            <Zap className="h-3 w-3 text-accent-400" />
            Explainable ML-powered analysis
          </div>

          <h1 className="font-heading text-5xl font-bold leading-tight tracking-tight text-white md:text-6xl">
            Understand how well
            <br />
            <span className="bg-gradient-to-r from-blue-300 via-accent-300 to-emerald-300 bg-clip-text text-transparent">
              your CV matches
            </span>{" "}
            the job
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-lg text-white/60">
            Upload your CV and a job description to receive an explainable
            compatibility analysis with matched skills, skill gaps, and
            actionable recommendations.
          </p>

          <div className="mt-10 flex items-center justify-center gap-4">
            <Link
              href="/analyze"
              className="group inline-flex items-center gap-2 rounded-xl bg-white px-7 py-3.5 font-heading text-sm font-semibold text-primary-700 shadow-lg shadow-black/10 transition-all duration-200 hover:bg-white/95 hover:shadow-xl hover:shadow-black/15 hover:-translate-y-0.5"
            >
              Get Started
              <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-0.5" />
            </Link>
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/5 px-7 py-3.5 font-heading text-sm font-semibold text-white/80 backdrop-blur-sm transition-all duration-200 hover:bg-white/10 hover:text-white hover:border-white/25"
            >
              View Dashboard
            </Link>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="bg-gradient-surface">
        <div className="mx-auto max-w-6xl px-6 py-20">
          <div className="mb-12 text-center">
            <h2 className="font-heading text-3xl font-bold tracking-tight text-gray-900">
              How It Works
            </h2>
            <p className="mt-3 text-gray-500">
              Three steps to an explainable compatibility report
            </p>
          </div>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <FeatureCard
              step="1"
              icon={<FileText className="h-5 w-5" />}
              title="Upload CV"
              description="Upload your PDF, DOCX, or TXT resume for structured parsing and skill extraction."
              color="blue"
            />
            <FeatureCard
              step="2"
              icon={<Brain className="h-5 w-5" />}
              title="Paste Job Description"
              description="Provide a job posting and the system extracts requirements, skills, and experience needs."
              color="emerald"
            />
            <FeatureCard
              step="3"
              icon={<BarChart3 className="h-5 w-5" />}
              title="Get Analysis"
              description="Receive a compatibility score, skill match breakdown, and clear recommendations."
              color="violet"
            />
          </div>
        </div>
      </section>

      {/* Trust bar */}
      <section className="border-t border-gray-200/60 bg-white/60 backdrop-blur-sm">
        <div className="mx-auto max-w-5xl px-6 py-12">
          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            <TrustItem
              icon={<Shield className="h-5 w-5" />}
              title="Privacy-First"
              description="No accounts needed. Your documents never leave your browser."
            />
            <TrustItem
              icon={<Brain className="h-5 w-5" />}
              title="Explainable AI"
              description="Every score comes with evidence. Understand why, not just what."
            />
            <TrustItem
              icon={<CheckCircle2 className="h-5 w-5" />}
              title="280+ Skills"
              description="Taxonomy across 26 career domains with semantic matching."
            />
          </div>
        </div>
      </section>
    </main>
  );
}

function FeatureCard({
  step,
  icon,
  title,
  description,
  color,
}: {
  step: string;
  icon: React.ReactNode;
  title: string;
  description: string;
  color: "blue" | "emerald" | "violet";
}) {
  const colorMap = {
    blue: "from-blue-500 to-primary-600",
    emerald: "from-accent-500 to-emerald-600",
    violet: "from-violet-500 to-purple-600",
  };

  const bgColorMap = {
    blue: "bg-blue-50",
    emerald: "bg-emerald-50",
    violet: "bg-violet-50",
  };

  return (
    <div className="card-premium-hover group p-6">
      <div className="mb-4 flex items-center gap-3">
        <div
          className={`flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br ${colorMap[color]} text-white shadow-lg transition-transform duration-300 group-hover:scale-105`}
        >
          {icon}
        </div>
        <span className="text-xs font-semibold uppercase tracking-widest text-gray-400">
          Step {step}
        </span>
      </div>
      <h4 className="font-heading text-lg font-semibold text-gray-900">
        {title}
      </h4>
      <p className="mt-2 text-sm leading-relaxed text-gray-500">
        {description}
      </p>
    </div>
  );
}

function TrustItem({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="flex items-start gap-4">
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary-700/8 text-primary-700">
        {icon}
      </div>
      <div>
        <h4 className="font-heading text-sm font-semibold text-gray-900">
          {title}
        </h4>
        <p className="mt-0.5 text-sm text-gray-500">{description}</p>
      </div>
    </div>
  );
}
