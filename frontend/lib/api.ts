/**
 * API client for the Career Match backend.
 *
 * Backend contract (backend/app/api/*):
 *   POST /api/matches        → MatchReport (raw-text mode: no persistence)
 *   GET  /api/history        → HistoryEntry[]
 *   GET  /api/matches/{id}   → MatchDetail
 *   GET  /health             → { status, version }
 */

import axios from "axios";
import type {
  HistoryEntry,
  MatchDetail,
  MatchReport,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export const api = axios.create({
  baseURL: API_URL,
  timeout: 120_000, // embedding inference on first request can be slow
  headers: { "Content-Type": "application/json" },
});

// --- Errors -----------------------------------------------------------------

export function apiErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = (err.response?.data as { detail?: unknown } | undefined)
      ?.detail;
    if (typeof detail === "string") return detail;
    if (detail !== undefined) return JSON.stringify(detail);
    if (err.code === "ECONNREFUSED")
      return "Cannot reach the analysis service. Is the backend running on port 8000?";
    return err.message;
  }
  return "An unexpected error occurred.";
}

// --- Health -----------------------------------------------------------------

export async function healthCheck(): Promise<{ status: string; version: string }> {
  const base = API_URL.replace(/\/api\/?$/, "");
  const { data } = await axios.get(`${base}/health`);
  return data;
}

// --- Matching ---------------------------------------------------------------

export interface MatchRequestBody {
  cv_text?: string;
  job_text?: string;
  resume_id?: number;
  job_id?: number;
  weights?: Record<string, number>;
}

export async function createMatch(body: MatchRequestBody): Promise<MatchReport> {
  const { data } = await api.post<MatchReport>("/matches", body);
  return data;
}

// --- History ----------------------------------------------------------------

export async function getHistory(limit = 50, offset = 0): Promise<HistoryEntry[]> {
  const { data } = await api.get<HistoryEntry[]>("/history", {
    params: { limit, offset },
  });
  return data;
}

export async function getMatchDetail(matchId: number): Promise<MatchDetail> {
  const { data } = await api.get<MatchDetail>(`/matches/${matchId}`);
  return data;
}
