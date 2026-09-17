"use client";

/**
 * /analyze — upload or paste a CV and a job description, get an
 * explainable compatibility report via POST /api/matches (raw-text mode).
 *
 * Privacy model (no accounts, by design): the analysis runs statelessly,
 * and the report is saved to THIS browser's localStorage only. The server
 * never persists the documents or the result. File uploads are extracted
 * server-side and immediately discarded — no copy is kept.
 */

import { useCallback, useRef, useState } from "react";
import Link from "next/link";
import { Upload, FileText, X, Check, ArrowRight } from "lucide-react";
import { createMatch, extractResume, apiErrorMessage } from "@/lib/api";
import { saveAnalysis } from "@/lib/history";
import type { MatchReport } from "@/lib/types";
import { Card, ErrorNote, MatchReportSkeleton } from "@/components/ui";
import { MatchReportView } from "@/components/MatchReportView";

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

const EXAMPLE_JOB = `Senior Platform Engineer — Helios Cloud

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
    <main className="mx-auto max-w-6xl px-6 py-10">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="font-heading text-2xl font-bold tracking-tight text-gray-900">
            Analyze a CV against a job
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Both documents are parsed, extracted, and matched — every score
            comes with its evidence.
          </p>
        </div>
        <Link
          href="/history"
          className="text-sm font-medium text-gray-500 transition-colors hover:text-primary-600"
        >
          History →
        </Link>
      </div>

      <form onSubmit={onSubmit} className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* ----- CV column: upload or paste ----- */}
        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <label
              htmlFor="cv"
              className="block text-sm font-semibold text-gray-700"
            >
              CV / resume
            </label>
            <span className="rounded-md bg-gray-100 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-gray-500">
              PDF · DOCX · TXT
            </span>
          </div>

          {/* Dropzone — hidden once a document has been extracted */}
          {upload.kind !== "done" ? (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
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
              className={`mb-3 flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-8 text-center transition-all duration-200 ${
                dragOver
                  ? "border-primary-500 bg-primary-50/50 shadow-glow-blue"
                  : "border-gray-200 hover:border-primary-300 hover:bg-primary-50/30 hover:shadow-glow-blue"
              }`}
            >
              <div
                className={`mb-3 flex h-12 w-12 items-center justify-center rounded-xl transition-colors duration-200 ${
                  dragOver
                    ? "bg-primary-100 text-primary-600"
                    : "bg-gray-100 text-gray-400"
                }`}
              >
                <Upload className="h-5 w-5" />
              </div>
              <p className="text-sm font-medium text-gray-700">
                {upload.kind === "uploading"
                  ? `Extracting ${upload.filename}…`
                  : "Drop a CV here, or click to browse"}
              </p>
              <p className="mt-1 text-xs text-gray-400">
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
            <div className="mb-3 flex items-center justify-between rounded-xl border border-accent-200/60 bg-gradient-to-r from-accent-50/80 to-emerald-50/60 px-4 py-3 shadow-sm">
              <span className="flex items-center gap-2.5 text-sm text-gray-700">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-100">
                  <FileText className="h-4 w-4 text-accent-600" />
                </div>
                <div>
                  <span className="font-medium">{upload.filename}</span>
                  <span className="ml-2 text-xs text-gray-400">
                    {upload.chars.toLocaleString()} chars
                  </span>
                </div>
                <Check className="h-4 w-4 text-accent-500" />
              </span>
              <button
                type="button"
                onClick={clearUpload}
                className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600"
              >
                <X className="h-3.5 w-3.5" /> clear
              </button>
            </div>
          )}

          {upload.kind === "error" ? (
            <p className="mb-3 text-xs font-medium text-red-600">
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
            placeholder="…or paste the full CV text here…"
            className="input-premium"
          />
          <button
            type="button"
            onClick={() => setCvText(EXAMPLE_CV)}
            className="mt-2 text-xs font-medium text-gray-400 underline decoration-gray-300 underline-offset-2 transition-colors hover:text-primary-600 hover:decoration-primary-300"
          >
            Use example CV
          </button>
        </div>

        {/* ----- Job column: paste only ----- */}
        <div>
          <label
            htmlFor="job"
            className="mb-1.5 block text-sm font-semibold text-gray-700"
          >
            Job description
          </label>
          <textarea
            id="job"
            value={jobText}
            onChange={(e) => setJobText(e.target.value)}
            rows={18}
            placeholder="Paste the job description here…"
            className="input-premium"
          />
          <button
            type="button"
            onClick={() => setJobText(EXAMPLE_JOB)}
            className="mt-2 text-xs font-medium text-gray-400 underline decoration-gray-300 underline-offset-2 transition-colors hover:text-primary-600 hover:decoration-primary-300"
          >
            Use example job
          </button>
        </div>

        <div className="lg:col-span-2">
          <button
            type="submit"
            disabled={!canSubmit}
            className="btn-primary"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                Analyzing…
              </span>
            ) : (
              <span className="flex items-center gap-2">
                Analyze match
                <ArrowRight className="h-4 w-4" />
              </span>
            )}
          </button>
          {!loading && !report && cvText.trim().length <= 40 ? (
            <p className="mt-2 text-xs text-gray-400">
              Upload or paste a CV (at least a few lines) to begin.
            </p>
          ) : null}
          {saved ? (
            <p className="ml-3 inline text-xs text-accent-600 font-medium">
              Saved to this browser&apos;s history.
            </p>
          ) : null}
        </div>
      </form>

      <div className="mt-12">
        {error ? <ErrorNote message={error} /> : null}
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
            title="Match report"
            subtitle="Live analysis — stored only in this browser"
          />
        ) : null}
      </div>
    </main>
  );
}
