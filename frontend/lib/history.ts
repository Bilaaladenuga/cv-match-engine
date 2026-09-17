/**
 * Browser-only analysis history.
 *
 * Privacy-first design (user decision: no accounts, free software):
 * every analysis is stored in THIS browser's localStorage and nowhere
 * else. The server keeps no copy — the matches endpoint is stateless.
 *
 * Shape: a full MatchReport (the exact POST /api/matches payload) plus
 * light metadata so lists and the detail page can render without
 * re-fetching anything.
 */

import type { MatchReport } from "./types";

const STORAGE_KEY = "career-match:history";
const MAX_ENTRIES = 100;

export interface StoredAnalysis {
  /** Server-assigned id when the match was persisted; local id otherwise. */
  id: string;
  /** ISO timestamp of when the analysis was run. */
  created_at: string;
  /** Free-form label shown in lists (defaults to the job's first line). */
  title: string;
  /** The original CV text that was submitted. */
  cv_text?: string;
  /** The original job description text that was submitted. */
  job_text?: string;
  report: MatchReport;
}

function readAll(): StoredAnalysis[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as StoredAnalysis[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    // Corrupted store — reset rather than crash the UI.
    window.localStorage.removeItem(STORAGE_KEY);
    return [];
  }
}

function writeAll(entries: StoredAnalysis[]): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
}

export function saveAnalysis(
  report: MatchReport,
  title?: string,
  cvText?: string,
  jobText?: string
): StoredAnalysis {
  const entry: StoredAnalysis = {
    id:
      report.match_id !== null && report.match_id !== undefined
        ? `srv-${report.match_id}`
        : `local-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    created_at: new Date().toISOString(),
    title: title?.trim() || `Analysis · ${report.overall_percent ?? "?"}%`,
    cv_text: cvText,
    job_text: jobText,
    report,
  };
  const entries = readAll();
  entries.unshift(entry); // newest first
  writeAll(entries.slice(0, MAX_ENTRIES));
  return entry;
}

export function listAnalyses(): StoredAnalysis[] {
  return readAll();
}

export function getAnalysis(id: string): StoredAnalysis | undefined {
  return readAll().find((e) => e.id === id);
}

export function deleteAnalysis(id: string): void {
  writeAll(readAll().filter((e) => e.id !== id));
}

export function clearHistory(): void {
  window.localStorage.removeItem(STORAGE_KEY);
}
