"use client";

import { useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Upload, FileText, X, Check, ArrowRight } from "lucide-react";
import { createMatch, extractResume, apiErrorMessage } from "@/lib/api";
import { saveAnalysis } from "@/lib/history";
import type { MatchReport } from "@/lib/types";
import { ErrorNote, MatchReportSkeleton } from "@/components/ui";
import { MatchReportView } from "@/components/MatchReportView";
import MobileNav from "@/components/MobileNav";

const EXAMPLE_CV = `Jane Okafor
Senior Backend Engineer
jane.okafor@example.com

SUMMARY
Backend engineer with 6 years building Python services and REST APIs.

SKILLS
Python, FastAPI, Django, PostgreSQL, Redis, Docker, Kubernetes, AWS, Git, CI/CD

EXPERIENCE
Senior Backend Engineer | Initech | 2020-03 - 2024-05
Built payment APIs with Python and FastAPI on PostgreSQL, deployed with Docker.
Software Engineer | Globex | 2018-01 - 2020-02
Django monolith to microservices, PostgreSQL schema design, Redis caching.

EDUCATION
BSc Computer Science, University of Leeds, 2017`;

const EXAMPLE_JOB = `Senior Platform Engineer at Helios Cloud

We need a platform engineer with 4+ years of experience.

Required skills: Python, Kubernetes, Terraform, AWS, CI/CD.
Preferred: Redis, observability tooling.

Responsibilities: own the Kubernetes platform, build CI/CD pipelines,
manage AWS infrastructure with Terraform.`;

type UploadState =
  | { kind: "idle" }
  | { kind: "uploading"; filename: string }
  | { kind: "done"; filename: string; chars: number }
  | { kind: "error"; filename: string; message: string };

export default function AnalyzePage() {
  const [cvText, setCvText] = useState("");
  const [jobText, setJobText] = useState("");
  const [report, setReport] = useState<MatchReport | null>(null);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [upload, setUpload] = useState<UploadState>({ kind: "idle" });
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const canSubmit =
    cvText.trim().length > 40 && jobText.trim().length > 40 && !loading;

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setReport(null);
    setSaved(false);
    try {
      const result = await createMatch({
        cv_text: cvText,
        job_text: jobText,
      });
      setReport(result);
      saveAnalysis(
        result,
        jobText.trim().split("\n", 1)[0]?.slice(0, 80) || undefined,
        cvText,
        jobText
      );
      setSaved(true);
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  const handleFile = useCallback(
    async (file: File) => {
      setUpload({ kind: "uploading", filename: file.name });
      try {
        const result = await extractResume(file);
        setCvText(result.text);
        setUpload({
          kind: "done",
          filename: result.filename,
          chars: result.char_count,
        });
      } catch (err) {
        setUpload({
          kind: "error",
          filename: file.name,
          message: apiErrorMessage(err),
        });
      }
    },
    []
  );

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) void handleFile(file);
  }

  function clearUpload() {
    setUpload({ kind: "idle" });
    setCvText("");
    if (fileInput.current) fileInput.current.value = "";
  }

  return (
    <main className="bg-canvas min-h-screen">
      {/* Nav */}
      <MobileNav />
      <nav className="fixed top-0 left-0 right-0 z-50 hidden items-center justify-center px-3 py-6 md:flex">
        <div className="flex items-center gap-4 rounded-pill bg-paper px-4 py-3 sm:gap-8 sm:px-8">
          <Link href="/" className="font-display text-lg uppercase tracking-tight text-carbon sm:text-xl">
            CV Match
          </Link>
          <div className="flex items-center gap-4 sm:gap-6">
            <Link href="/analyze" className="text-body-sm font-medium text-carbon sm:text-body">
              Analyze
            </Link>
            <Link href="/history" className="text-body-sm font-medium text-slate hover:text-carbon transition-colors sm:text-body">
              History
            </Link>
          </div>
        </div>
      </nav>

      <div className="max-w-page mx-auto px-4 sm:px-8 pt-32 pb-section">
        {/* Header */}
        <div className="mb-10 sm:mb-12">
          <span className="tag mb-4 inline-block">ANALYSIS</span>
          <h1 className="heading-display text-display text-carbon mb-2">
            Match Your CV
          </h1>
          <p className="text-body text-slate max-w-lg">
            Paste your CV and the job description. Every score comes with
            evidence so you know exactly what to fix.
          </p>
        </div>

        <form onSubmit={onSubmit} className="grid grid-cols-1 gap-6 lg:grid-cols-2 lg:gap-8">
          {/* CV Column */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <label htmlFor="cv" className="font-body text-sub font-medium uppercase text-carbon">
                Your CV
              </label>
              <span className="label-mono">PDF / DOCX / TXT</span>
            </div>

            {upload.kind !== "done" ? (
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => fileInput.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    fileInput.current?.click();
                  }
                }}
                className={`mb-4 flex cursor-pointer flex-col items-center justify-center rounded-card border-2 border-dashed px-4 py-10 text-center transition-all ${
                  dragOver
                    ? "border-carbon bg-mist"
                    : "border-ash hover:border-carbon"
                }`}
              >
                <div className="mb-3 h-12 w-12 rounded-lg bg-carbon text-paper flex items-center justify-center">
                  <Upload className="h-5 w-5" />
                </div>
                <p className="text-body font-medium text-carbon">
                  {upload.kind === "uploading"
                    ? `Reading ${upload.filename}...`
                    : "Drop a CV here or click to browse"}
                </p>
                <p className="label-mono mt-1">
                  The file is read once and never stored.
                </p>
                <input
                  ref={fileInput}
                  type="file"
                  accept=".pdf,.docx,.txt"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void handleFile(file);
                  }}
                />
              </div>
            ) : (
              <div className="mb-4 flex items-center justify-between rounded-card border border-mint bg-mint/30 px-4 py-3">
                <span className="flex items-center gap-3 text-body text-carbon">
                  <div className="h-8 w-8 rounded-lg bg-carbon text-paper flex items-center justify-center">
                    <FileText className="h-4 w-4" />
                  </div>
                  <div>
                    <span className="font-medium">{upload.filename}</span>
                    <span className="label-mono ml-2">{upload.chars.toLocaleString()} chars</span>
                  </div>
                  <Check className="h-4 w-4 text-carbon" />
                </span>
                <button
                  type="button"
                  onClick={clearUpload}
                  className="flex items-center gap-1 rounded-lg px-2 py-1 text-caption text-smoke hover:bg-ash/30 transition-colors"
                >
                  <X className="h-3.5 w-3.5" /> clear
                </button>
              </div>
            )}

            {upload.kind === "error" ? (
              <p className="mb-3 text-body-sm font-medium text-carbon bg-voltage px-3 py-2 rounded-lg">
                {upload.message}
              </p>
            ) : null}

            <textarea
              id="cv"
              value={cvText}
              onChange={(e) => {
                setCvText(e.target.value);
                if (upload.kind === "done") setUpload({ kind: "idle" });
              }}
              rows={14}
              placeholder="Or paste your CV text here..."
              className="input"
            />
            <button
              type="button"
              onClick={() => setCvText(EXAMPLE_CV)}
              className="mt-3 label-mono text-smoke hover:text-carbon transition-colors underline underline-offset-2"
            >
              Use example CV
            </button>
          </div>

          {/* Job Column */}
          <div>
            <label htmlFor="job" className="font-body text-sub font-medium uppercase text-carbon block mb-3">
              Job Description
            </label>
            <textarea
              id="job"
              value={jobText}
              onChange={(e) => setJobText(e.target.value)}
              rows={18}
              placeholder="Paste the job description here..."
              className="input"
            />
            <button
              type="button"
              onClick={() => setJobText(EXAMPLE_JOB)}
              className="mt-3 label-mono text-smoke hover:text-carbon transition-colors underline underline-offset-2"
            >
              Use example job
            </button>
          </div>

          {/* Submit */}
          <div className="lg:col-span-2">
            <button
              type="submit"
              disabled={!canSubmit}
              className="btn-primary"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-ash border-t-carbon" />
                  Analyzing...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  Analyze Match
                  <ArrowRight className="h-4 w-4" />
                </span>
              )}
            </button>
            {!loading && !report && cvText.trim().length <= 40 ? (
              <p className="mt-3 label-mono text-smoke">
                Paste a CV (at least a few lines) to get started.
              </p>
            ) : null}
            {saved ? (
              <p className="ml-4 inline label-mono text-carbon font-medium">
                Saved to browser history.
              </p>
            ) : null}
          </div>
        </form>

        {/* Results */}
        <div className="mt-16">
          {error ? (
            <div className="card bg-voltage/20 border border-voltage">
              <ErrorNote message={error} />
            </div>
          ) : null}
          {loading ? <MatchReportSkeleton /> : null}
          {report ? (
            <MatchReportView
              report={report}
              lists={{
                matched: (report.skill_evidence ?? [])
                  .filter((e) => e.status === "matched")
                  .map((e) => e.skill),
                partial: (report.skill_evidence ?? [])
                  .filter((e) => e.status === "partial")
                  .map((e) => e.skill),
                missing: (report.skill_evidence ?? [])
                  .filter((e) => e.status === "missing")
                  .map((e) => e.skill),
              }}
              title="Match Report"
              subtitle="Stored only in this browser"
              exportTexts={{ cvText, jobText }}
            />
          ) : null}
        </div>
      </div>
    </main>
  );
}
