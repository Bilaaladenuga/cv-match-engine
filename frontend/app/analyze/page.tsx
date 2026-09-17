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
import { Upload, FileText, X, Check } from "lucide-react";
import { createMatch, extractResume, apiErrorMessage } from "@/lib/api";
import { saveAnalysis } from "@/lib/history";
import type { MatchReport } from "@/lib/types";
import { Card, ErrorNote, Loading } from "@/components/ui";
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
        jobText.trim().split("\n", 1)[0]?.slice(0, 80) || undefined
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
          <h1 className="text-2xl font-semibold text-gray-900">
            Analyze a CV against a job
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Both documents are parsed, extracted, and matched — every score
            comes with its evidence.
          </p>
        </div>
        <Link
          href="/history"
          className="text-sm text-gray-500 hover:text-gray-900"
        >
          History →
        </Link>
      </div>

      <form onSubmit={onSubmit} className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* ----- CV column: upload or paste ----- */}
        <div>
          <div className="mb-1 flex items-center justify-between">
            <label
              htmlFor="cv"
              className="block text-sm font-medium text-gray-700"
            >
              CV / resume
            </label>
            <span className="text-xs text-gray-400">PDF · DOCX · TXT</span>
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
              className={`mb-2 flex cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed px-4 py-6 text-center transition-colors ${
                dragOver
                  ? "border-gray-500 bg-gray-50"
                  : "border-gray-300 hover:border-gray-400 hover:bg-gray-50"
              }`}
            >
              <Upload className="mb-1 h-5 w-5 text-gray-400" />
              <p className="text-sm text-gray-600">
                {upload.kind === "uploading"
                  ? `Extracting ${upload.filename}…`
                  : "Drop a CV here, or click to browse"}
              </p>
              <p className="mt-0.5 text-xs text-gray-400">
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
            <div className="mb-2 flex items-center justify-between rounded-md border border-gray-200 bg-gray-50 px-3 py-2">
              <span className="flex items-center gap-2 text-sm text-gray-700">
                <FileText className="h-4 w-4 text-gray-500" />
                {upload.filename}
                <span className="text-xs text-gray-400">
                  {upload.chars.toLocaleString()} chars
                </span>
                <Check className="h-4 w-4 text-green-600" />
              </span>
              <button
                type="button"
                onClick={clearUpload}
                className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600"
              >
                <X className="h-3.5 w-3.5" /> clear
              </button>
            </div>
          )}

          {upload.kind === "error" ? (
            <p className="mb-2 text-xs text-red-600">{upload.message}</p>
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
            className="w-full rounded-md border border-gray-300 px-3 py-2 font-mono text-sm text-gray-800 placeholder:text-gray-400 focus:border-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-500"
          />
          <button
            type="button"
            onClick={() => setCvText(EXAMPLE_CV)}
            className="mt-1 text-xs text-gray-500 underline hover:text-gray-700"
          >
            Use example CV
          </button>
        </div>

        {/* ----- Job column: paste only ----- */}
        <div>
          <label
            htmlFor="job"
            className="mb-1 block text-sm font-medium text-gray-700"
          >
            Job description
          </label>
          <textarea
            id="job"
            value={jobText}
            onChange={(e) => setJobText(e.target.value)}
            rows={18}
            placeholder="Paste the job description here…"
            className="w-full rounded-md border border-gray-300 px-3 py-2 font-mono text-sm text-gray-800 placeholder:text-gray-400 focus:border-gray-500 focus:outline-none focus:ring-1 focus:ring-gray-500"
          />
          <button
            type="button"
            onClick={() => setJobText(EXAMPLE_JOB)}
            className="mt-1 text-xs text-gray-500 underline hover:text-gray-700"
          >
            Use example job
          </button>
        </div>

        <div className="lg:col-span-2">
          <button
            type="submit"
            disabled={!canSubmit}
            className="rounded-md bg-gray-900 px-5 py-2.5 text-sm font-medium text-white hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-300"
          >
            {loading ? "Analyzing…" : "Analyze match"}
          </button>
          {!loading && !report && cvText.trim().length <= 40 ? (
            <p className="mt-2 text-xs text-gray-400">
              Upload or paste a CV (at least a few lines) to begin.
            </p>
          ) : null}
          {saved ? (
            <p className="ml-3 inline text-xs text-gray-500">
              Saved to this browser&apos;s history.
            </p>
          ) : null}
        </div>
      </form>

      <div className="mt-10">
        {error ? <ErrorNote message={error} /> : null}
        {loading ? <Loading /> : null}
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
