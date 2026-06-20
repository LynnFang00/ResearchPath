export type Recommendation = {
  paper_id: number;
  title: string;
  abstract_snippet: string;
  year: number | null;
  authors: string[];
  score: number;
  method: string;
  explanation: string;
};

export type PathPaper = Recommendation & {
  difficulty_label: string;
  difficulty_score: number;
  difficulty_explanation: string;
  path_section: string;
  path_reason: string;
  relevance_score?: number | null;
  bm25_score?: number | null;
  tfidf_score?: number | null;
  faiss_score?: number | null;
  citation_score?: number | null;
  influence_score?: number | null;
  recency_score?: number | null;
  difficulty_fit_score?: number | null;
  background_signal?: number | null;
  method_signal?: number | null;
  narrow_application_score?: number | null;
  duplicate_penalty?: number | null;
  too_narrow_penalty?: number | null;
  final_path_score?: number | null;
};

export type ReadingPath = {
  query: string;
  method: string;
  background_level: string;
  sections: Record<string, PathPaper[]>;
  section_status: Record<string, { section_complete: boolean; fill_reason: string | null }>;
};

export type DatasetStatus = {
  dataset_name: string;
  source: string;
  paper_count: number;
  citation_edge_count: number;
  last_updated_timestamp: string | null;
  model_index_version: string;
  embedding_model_name: string;
  faiss_index_path: string;
  manifest_path: string | null;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

export async function searchRecommendations(query: string, k = 10, method = 'hybrid'): Promise<Recommendation[]> {
  const params = new URLSearchParams({ query, k: String(k), method });
  const response = await fetch(`${API_BASE_URL}/recommend/query?${params.toString()}`);

  if (!response.ok) {
    throw new Error(`Search failed with status ${response.status}`);
  }

  return response.json();
}

export async function fetchReadingPath(
  query: string,
  k = 4,
  method = 'hybrid',
  backgroundLevel = 'basic_ml',
): Promise<ReadingPath> {
  const params = new URLSearchParams({
    query,
    k: String(k),
    method,
    background_level: backgroundLevel,
  });
  const response = await fetch(`${API_BASE_URL}/path/query?${params.toString()}`);

  if (!response.ok) {
    throw new Error(`Reading path failed with status ${response.status}`);
  }

  return response.json();
}

export async function fetchDatasetStatus(): Promise<DatasetStatus> {
  const response = await fetch(`${API_BASE_URL}/dataset/status`);

  if (!response.ok) {
    throw new Error(`Dataset status failed with status ${response.status}`);
  }

  return response.json();
}
