"use client";

/**
 * /analyze — paste a CV and a job description, get an explainable
 * compatibility report via POST /api/matches (raw-text mode).
 *
 * Raw-text mode is stateless: nothing is persisted until auth exists.
 * File upload (PDF/DOCX) is a later slice — the backend parsers are
 * ready but the upload endpoint is not yet wired.
 */

import { useState } from "react";
import Link from "next/link";
import { createMatch, apiErrorMessage } from "@/lib/api";
import type { MatchReport } from "@/lib/types";
import { ErrorNote, Loading } from "@/components/ui";
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

export default function AnalyzePage() {
  const [cvText, setCvText] = useState("");
  const [jobText, setJobText] = useState("");
  const [report, setReport] = useState<MatchReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canSubmit =
    cvText.trim().length > 40 && jobText.trim().length > 40 && !loading;

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const result = await createMatch({
        cv_text: cvText,
        job_text: jobText,
      });
      setReport(result);
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setLoading(false);
    }
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
        <div>
          <label
            htmlFor="cv"
            className="mb-1 block text-sm font-medium text-gray-700"
          >
            CV / resume text
          </label>
          <textarea
            id="cv"
            value={cvText}
            onChange={(e) => setCvText(e.target.value)}
            rows={18}
            placeholder="Paste the full CV text here…"
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
              Paste a CV (at least a few lines) to begin.
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
            subtitle="Live analysis — not saved (no account system yet)"
          />
        ) : null}
      </div>
    </main>
  );
}
