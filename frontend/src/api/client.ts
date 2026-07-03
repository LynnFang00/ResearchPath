export type Recommendation = {
  paper_id: number;
  title: string;
  abstract_snippet: string;
  year: number | null;
  authors: string[];
  venue: string | null;
  paper_url: string | null;
  pdf_url: string | null;
  doi_url: string | null;
  source_url: string | null;
  doi: string | null;
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
  why_recommended?: string | null;
  why_this_section?: string | null;
  confidence_label?: string | null;
  read_before?: string[];
  read_after?: string[];
  explanation_signals?: string[];
  paper_type_tags?: string[];
  personalization_score?: number | null;
  personalization_reason?: string | null;
  saved_similarity?: number | null;
  skipped_similarity?: number | null;
  too_hard_similarity?: number | null;
  topic_similarity?: number | null;
  learned_ranker_score?: number | null;
  learned_ranker_adjustment?: number | null;
  learned_ranker_version?: string | null;
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

export type FeedbackAction =
  | 'save'
  | 'already_read'
  | 'too_easy'
  | 'too_hard'
  | 'not_relevant'
  | 'more_like_this'
  | 'less_like_this';

export type FeedbackPayload = {
  paper_id: number;
  query: string;
  section?: string | null;
  action: FeedbackAction;
  method: string;
  background_level: string;
  tags?: string[];
};

export type UserProfile = {
  background_level: string;
  saved_paper_ids: number[];
  skipped_paper_ids: number[];
  too_easy_paper_ids: number[];
  too_hard_paper_ids: number[];
  preferred_topics: string[];
  avoid_topics: string[];
  current_status: string;
  research_goal: string;
  paper_taste: string;
  updated_at: string | null;
};

export type LibraryItem = {
  id: number;
  paper_id: number;
  tags: string[];
  notes: string;
  created_at: string | null;
  updated_at: string | null;
  paper: Recommendation;
};

export type LibraryResponse = {
  items: LibraryItem[];
  tags: string[];
};

export type MethodComparisonResult = {
  method: string;
  recommendations: Recommendation[];
  error?: string;
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

export async function fetchRecommendationMethods(): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/recommend/methods`);

  if (!response.ok) {
    throw new Error(`Method list failed with status ${response.status}`);
  }

  const payload = (await response.json()) as { methods?: string[] };
  return payload.methods ?? [];
}

export async function compareRecommendationMethods(
  query: string,
  k = 5,
  methods: string[],
): Promise<MethodComparisonResult[]> {
  return Promise.all(
    methods.map(async (method) => {
      try {
        return { method, recommendations: await searchRecommendations(query, k, method) };
      } catch (comparisonError) {
        return {
          method,
          recommendations: [],
          error: comparisonError instanceof Error ? comparisonError.message : 'Comparison failed.',
        };
      }
    }),
  );
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

export async function postFeedback(payload: FeedbackPayload): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Feedback failed with status ${response.status}`);
  }
}

export async function fetchProfile(): Promise<UserProfile> {
  const response = await fetch(`${API_BASE_URL}/profile`);

  if (!response.ok) {
    throw new Error(`Profile failed with status ${response.status}`);
  }

  return response.json();
}

export async function patchProfile(payload: Partial<UserProfile>): Promise<UserProfile> {
  const response = await fetch(`${API_BASE_URL}/profile`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Profile update failed with status ${response.status}`);
  }

  return response.json();
}

export async function fetchLibrary(tag?: string | null): Promise<LibraryResponse> {
  const params = new URLSearchParams();
  if (tag) params.set('tag', tag);
  const query = params.toString();
  const response = await fetch(`${API_BASE_URL}/library${query ? `?${query}` : ''}`);

  if (!response.ok) {
    throw new Error(`Library failed with status ${response.status}`);
  }

  return response.json();
}

export async function saveLibraryItem(paperId: number, tags: string[]): Promise<LibraryItem> {
  const response = await fetch(`${API_BASE_URL}/library/items`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper_id: paperId, tags }),
  });

  if (!response.ok) {
    throw new Error(`Library save failed with status ${response.status}`);
  }

  return response.json();
}

export async function deleteLibraryItem(paperId: number): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/library/items/${paperId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Library unsave failed with status ${response.status}`);
  }
}
