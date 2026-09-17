/**
 * TypeScript mirrors of the backend API schemas
 * (backend/app/schemas/match.py, backend/app/scoring/improvement_engine.py).
 * Keep in sync with the Pydantic models.
 */

export interface ComponentScore {
  name: string;
  raw_score: number; // 0..1
  weight: number;
  weighted: number; // raw_score * weight
  evidence: string;
}

/** One graded skill from the Phase 16 improvement engine. */
export interface SkillEvidence {
  skill: string;
  status: "matched" | "partial" | "missing" | "unknown";
  candidate_skill: string | null;
  strength: "strong" | "moderate" | "weak" | "absent";
  in_skills_section: boolean;
  in_work_history: boolean;
  estimated_months: number | null;
  sources: string[];
}

export interface MLDetails {
  label?: string;
  fit_score?: number;
  probabilities?: Record<string, number>;
  raw_probabilities?: Record<string, number>;
  calibration_method?: string | null;
  ml_version?: string;
  explanation?: {
    method?: string;
    factors?: Array<{
      feature: string;
      label: string;
      contribution: number;
      direction: "helps" | "hurts" | "neutral";
      value: number;
      reference: number;
      detail: string;
    }>;
    cautions?: string[];
    disclaimer?: string;
  };
  [key: string]: unknown;
}

/** Response for POST /api/matches (raw-text mode has match_id = null). */
export interface MatchReport {
  match_id: number | null;
  model_version: string;
  overall_score: number; // 0..1
  overall_percent: number; // 0..100
  band: string;
  components: ComponentScore[];
  weights: Record<string, number>;
  positive_factors: string[];
  negative_factors: string[];
  recommendations: string[];
  disclaimer: string;
  skill_matches?: Record<string, unknown>[];
  experience?: Record<string, unknown> | null;
  semantic?: Record<string, unknown> | null;
  education?: Record<string, unknown> | null;
  certifications?: Record<string, unknown> | null;
  skill_evidence?: SkillEvidence[];
  ml_details?: MLDetails | null;
}

/** One row of GET /api/history. */
export interface HistoryEntry {
  match_id: number;
  candidate_id: number;
  job_id: number;
  candidate_name: string | null;
  job_title: string | null;
  overall_score: number; // 0..1
  band: string | null;
  model_version: string;
  created_at: string | null;
  ml_fit_score: number | null;
  ml_label: string | null;
  matched_skills: string[];
  missing_skills: string[];
}

/** Response for GET /api/matches/{id}. */
export interface MatchDetail {
  match_id: number;
  candidate_id: number;
  job_id: number;
  candidate_name: string | null;
  job_title: string | null;
  job_company: string | null;
  overall_score: number;
  band: string | null;
  model_version: string;
  created_at: string | null;
  component_scores: Record<string, number>;
  ml: {
    fit_score: number | null;
    label: string | null;
    probabilities: Record<string, number> | null;
    calibration_method: string | null;
  };
  explanation: {
    matched_skills: string[];
    missing_skills: string[];
    partial_skills: string[];
    recommendations: string[];
    positive_factors: string[];
    negative_factors: string[];
  };
  feature_values: Record<string, unknown> | null;
}
