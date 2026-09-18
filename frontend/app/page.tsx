import Link from "next/link";
import {
  FileText,
  Brain,
  Shield,
  CheckCircle2,
  ArrowRight,
  Upload,
  BarChart3,
  Lightbulb,
  Zap,
} from "lucide-react";
import {
  FadeIn,
  SlideUp,
  StaggerContainer,
  StaggerItem,
  ScrollReveal,
  CountUp,
} from "@/components/motion";

export default function HomePage() {
  return (
    <main className="bg-canvas min-h-screen">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-center py-6">
        <div className="flex items-center gap-8 rounded-pill bg-paper px-8 py-3">
          <span className="font-display text-xl uppercase tracking-tight text-carbon">
            CV Match
          </span>
          <div className="flex items-center gap-6">
            <Link
              href="/analyze"
              className="text-body font-medium text-slate hover:text-carbon transition-colors"
            >
              Analyze
            </Link>
            <Link
              href="/history"
              className="text-body font-medium text-slate hover:text-carbon transition-colors"
            >
              History
            </Link>
            <Link href="/analyze" className="btn-primary text-sm">
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="px-8 pt-40 pb-section max-w-page mx-auto">
        <div className="grid grid-cols-2 gap-16 items-center">
          <div>
            <FadeIn delay={0.1}>
              <div className="tag mb-6">AI-Powered Analysis</div>
            </FadeIn>
            <FadeIn delay={0.2}>
              <h1 className="heading-display text-display-xl text-carbon mb-6">
                Know Your
                <br />
                CV Score
                <br />
                Before They
                <br />
                Do
              </h1>
            </FadeIn>
            <FadeIn delay={0.3}>
              <p className="text-body text-slate mb-8 max-w-md leading-relaxed">
                Drop in your CV and a job description. You will get a score
                with clear feedback on what to fix and why.
              </p>
            </FadeIn>
            <FadeIn delay={0.4}>
              <div className="flex items-center gap-4">
                <Link href="/analyze" className="btn-primary">
                  Start Analyzing
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
                <Link href="#how-it-works" className="btn-ghost">
                  How It Works
                </Link>
              </div>
            </FadeIn>
          </div>
          <FadeIn delay={0.3} className="relative">
            <div className="card-inverted p-12">
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <span className="label-mono">COMPATIBILITY</span>
                  <span className="tag">87%</span>
                </div>
                <div className="h-2 rounded-full bg-graphite overflow-hidden">
                  <div className="h-full w-[87%] bg-mint rounded-full" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-graphite rounded-card p-4">
                    <span className="label-mono block mb-1">SKILLS</span>
                    <span className="font-display text-heading-lg text-paper">92%</span>
                  </div>
                  <div className="bg-graphite rounded-card p-4">
                    <span className="label-mono block mb-1">EXPERIENCE</span>
                    <span className="font-display text-heading-lg text-paper">85%</span>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-smoke text-body-sm">
                  <CheckCircle2 className="h-4 w-4 text-mint" />
                  <span>Strong match for Senior Backend Engineer</span>
                </div>
              </div>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="bg-carbon py-12">
        <div className="max-w-page mx-auto px-8 grid grid-cols-4 gap-8">
          <ScrollReveal>
            <div>
              <span className="font-display text-display text-paper">
                <CountUp target={280} suffix="+" />
              </span>
              <p className="label-mono text-smoke mt-1">Skills Tracked</p>
            </div>
          </ScrollReveal>
          <ScrollReveal delay={0.1}>
            <div>
              <span className="font-display text-display text-paper">
                <CountUp target={26} />
              </span>
              <p className="label-mono text-smoke mt-1">Career Domains</p>
            </div>
          </ScrollReveal>
          <ScrollReveal delay={0.2}>
            <div>
              <span className="font-display text-display text-paper">
                <CountUp target={49} />
              </span>
              <p className="label-mono text-smoke mt-1">ML Features</p>
            </div>
          </ScrollReveal>
          <ScrollReveal delay={0.3}>
            <div>
              <span className="font-display text-display text-paper">
                <CountUp target={100} suffix="%" />
              </span>
              <p className="label-mono text-smoke mt-1">Private & Local</p>
            </div>
          </ScrollReveal>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-section max-w-page mx-auto px-8">
        <ScrollReveal>
          <div className="mb-12">
            <span className="label-mono block mb-2">PROCESS</span>
            <h2 className="heading-display text-display text-carbon">
              How It Works
            </h2>
          </div>
        </ScrollReveal>
        <StaggerContainer className="grid grid-cols-3 gap-8">
          <StaggerItem>
            <StepCard
              number="01"
              icon={<Upload className="h-6 w-6" />}
              title="Upload Documents"
              description="Paste your CV and the job description. We never store your files. Analysis happens on the fly."
            />
          </StaggerItem>
          <StaggerItem>
            <StepCard
              number="02"
              icon={<Brain className="h-6 w-6" />}
              title="AI Analysis"
              description="49 machine learning features scan for skill gaps, experience fit, and how well you match the role."
            />
          </StaggerItem>
          <StaggerItem>
            <StepCard
              number="03"
              icon={<Lightbulb className="h-6 w-6" />}
              title="Get Evidence"
              description="A scored breakdown with specific improvements, ranked by how much each one moves the needle."
            />
          </StaggerItem>
        </StaggerContainer>
      </section>

      {/* Features Section */}
      <section className="bg-carbon py-section">
        <div className="max-w-page mx-auto px-8">
          <ScrollReveal>
            <div className="mb-12">
              <span className="label-mono block mb-2 text-smoke">FEATURES</span>
              <h2 className="heading-display text-display text-paper">
                Built For Accuracy
              </h2>
            </div>
          </ScrollReveal>
          <StaggerContainer className="grid grid-cols-2 gap-8">
            <StaggerItem>
              <FeatureCard
                icon={<BarChart3 className="h-6 w-6" />}
                title="Ensemble ML Model"
                description="Three models work together: Gradient Boosting, Random Forest, and Logistic Regression, trained on real job postings."
              />
            </StaggerItem>
            <StaggerItem>
              <FeatureCard
                icon={<FileText className="h-6 w-6" />}
                title="ATS Friendliness"
                description="8-point check on section headings, contact info, skills, date format, length, formatting, keywords, and bullet points."
              />
            </StaggerItem>
            <StaggerItem>
              <FeatureCard
                icon={<Shield className="h-6 w-6" />}
                title="Privacy First"
                description="No sign-ups. No file storage. Your documents are processed and gone. Nothing leaves your browser."
              />
            </StaggerItem>
            <StaggerItem>
              <FeatureCard
                icon={<Zap className="h-6 w-6" />}
                title="Calibrated Scores"
                description="Platt scaling keeps predictions accurate. Error rate dropped from 15% down to 3%."
              />
            </StaggerItem>
          </StaggerContainer>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-section max-w-page mx-auto px-8">
        <ScrollReveal>
          <div className="card-inverted p-16 text-center">
            <h2 className="heading-display text-heading-lg text-paper mb-4">
              Stop Guessing.<br />Start Matching.
            </h2>
            <p className="text-body text-smoke mb-8 max-w-lg mx-auto">
              Get a scored breakdown in seconds. Know exactly what to fix
              before you hit apply.
            </p>
            <Link href="/analyze" className="btn-primary bg-paper text-carbon hover:bg-mint">
              Analyze Your CV
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </div>
        </ScrollReveal>
      </section>

      {/* Footer */}
      <footer className="bg-carbon py-12">
        <div className="max-w-page mx-auto px-8 flex items-center justify-between">
          <span className="font-display text-lg uppercase text-paper">
            CV Match Engine
          </span>
          <div className="flex items-center gap-6">
            <span className="label-mono text-smoke">
              Privacy-first. No accounts needed.
            </span>
          </div>
        </div>
      </footer>
    </main>
  );
}

function StepCard({
  number,
  icon,
  title,
  description,
}: {
  number: string;
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="card card-hover">
      <span className="label-mono text-ash block mb-4">STEP {number}</span>
      <div className="h-12 w-12 rounded-lg bg-carbon text-paper flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="font-body text-heading-sm font-medium uppercase text-carbon mb-2">
        {title}
      </h3>
      <p className="text-body text-slate leading-relaxed">{description}</p>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="bg-graphite rounded-card p-6">
      <div className="h-12 w-12 rounded-lg bg-mint text-carbon flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="font-body text-sub-lg font-medium uppercase text-paper mb-2">
        {title}
      </h3>
      <p className="text-body text-smoke leading-relaxed">{description}</p>
    </div>
  );
}
