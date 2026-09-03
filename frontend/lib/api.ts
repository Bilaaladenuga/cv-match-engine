/**
 * API client for the Career Match backend.
 */

import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// --- Types ---

export interface HealthResponse {
  status: string;
  version: string;
}

export interface ResumeUploadResponse {
  id: number;
  filename: string;
  status: string;
}

export interface JobResponse {
  id: number;
  title: string;
  company: string | null;
}

export interface MatchResult {
  id: number;
  overall_score: number;
  semantic_score: number;
  skills_score: number;
  experience_score: number;
  education_score: number;
  matched_skills: string[];
  missing_skills: string[];
  partial_skills: string[];
  recommendations: string[];
}

// --- API Calls ---

export async function healthCheck(): Promise<HealthResponse> {
  const { data } = await axios.get(`${API_URL.replace("/api", "")}/health`);
  return data;
}

export async function uploadResume(file: File): Promise<ResumeUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await api.post("/resumes/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function createJob(payload: {
  title: string;
  description: string;
  company?: string;
}): Promise<JobResponse> {
  const { data } = await api.post("/jobs", payload);
  return data;
}

export async function createMatch(payload: {
  resume_id: number;
  job_id: number;
}): Promise<MatchResult> {
  const { data } = await api.post("/matches", payload);
  return data;
}

export async function getMatch(matchId: number): Promise<MatchResult> {
  const { data } = await api.get(`/matches/${matchId}`);
  return data;
}
