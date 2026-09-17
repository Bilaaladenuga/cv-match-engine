export default function HomePage() {
  return (
    <main className="min-h-screen">
      {/* Hero */}
      <section className="mx-auto max-w-4xl px-6 py-20 text-center">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">
          Understand how well your CV matches the job
        </h2>
        <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
          Upload your CV and a job description to receive an explainable
          compatibility analysis with matched skills, skill gaps, and
          actionable recommendations.
        </p>
        <div className="flex gap-4 justify-center">
          <a
            href="/analyze"
            className="inline-block bg-gray-900 text-white px-6 py-3 rounded-md text-sm font-medium hover:bg-gray-800 transition-colors"
          >
            Get Started
          </a>
          <a
            href="/dashboard"
            className="inline-block border border-gray-300 text-gray-700 px-6 py-3 rounded-md text-sm font-medium hover:bg-gray-50 transition-colors"
          >
            View Dashboard
          </a>
        </div>
      </section>

      {/* Features */}
      <section className="bg-gray-50 border-t border-gray-200">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <h3 className="text-2xl font-semibold text-gray-900 mb-8 text-center">
            How It Works
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <FeatureCard
              step="1"
              title="Upload CV"
              description="Upload your PDF, DOCX, or TXT resume for structured parsing and skill extraction."
            />
            <FeatureCard
              step="2"
              title="Paste Job Description"
              description="Provide a job posting and the system extracts requirements, skills, and experience needs."
            />
            <FeatureCard
              step="3"
              title="Get Analysis"
              description="Receive a compatibility score, skill match breakdown, and clear recommendations."
            />
          </div>
        </div>
      </section>
    </main>
  );
}

function FeatureCard({
  step,
  title,
  description,
}: {
  step: string;
  title: string;
  description: string;
}) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6">
      <div className="text-sm font-medium text-gray-500 mb-2">Step {step}</div>
      <h4 className="text-lg font-semibold text-gray-900 mb-2">{title}</h4>
      <p className="text-sm text-gray-600">{description}</p>
    </div>
  );
}
