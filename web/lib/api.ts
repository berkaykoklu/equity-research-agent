/**
 * Types mirroring the Python note schema, and the two reads the site makes.
 *
 * These are the only place the frontend restates something the backend already
 * knows, and they are deliberately kept to shape alone. Anything with a rule
 * in it -- how an accession becomes an SEC URL, whether a claim verified, what
 * counts as coverage -- is computed in Python, under the test suite that gates
 * every change, and arrives here already decided.
 */

export type ChunkRef = { accession: string; chunk_id: number };

export type FactRef = {
  tag: string;
  fiscal_period: string;
  value: number;
  accession: string;
};

export type Claim = {
  text: string;
  chunks: ChunkRef[];
  facts: FactRef[];
};

export type Section = {
  name: string;
  claims: Claim[];
  available: boolean;
  unavailable_reason: string | null;
};

export type Coverage = {
  sections_available: number;
  sections_total: number;
  metrics_resolved: string[];
  metrics_missing: string[];
};

export type ResearchNote = {
  ticker: string;
  cik: string;
  sections: Section[];
  coverage: Coverage;
  accessions: string[];
  cost_usd: number;
  latency_seconds: number;
};

export type NoteSummary = {
  ticker: string;
  cik: string;
  generated_at: string;
  sections_available: number;
  sections_total: number;
};

export type FilingLink = { accession: string; url: string };

export type NoteDetail = NoteSummary & {
  note: ResearchNote;
  markdown: string;
  filings: FilingLink[];
};

/**
 * Where the read API lives.
 *
 * Same origin by default, which is what a single-project deployment gives us.
 * `API_BASE_URL` overrides it for the split-project layout and for local work,
 * where Next runs on 3000 and uvicorn on 8000.
 */
function baseUrl(): string {
  const configured = process.env.API_BASE_URL;
  if (configured) return configured.replace(/\/$/, "");
  if (process.env.VERCEL_URL) return `https://${process.env.VERCEL_URL}`;
  return "http://127.0.0.1:8000";
}

async function read<T>(path: string): Promise<T | null> {
  const response = await fetch(`${baseUrl()}${path}`, {
    // Notes change only when a note is deliberately regenerated, so serving a
    // slightly stale page is fine and revalidating on every view is not: the
    // database is a free tier that suspends when idle, and a cold wake is
    // several seconds a reader would spend staring at nothing.
    next: { revalidate: 300 },
  });

  if (response.status === 404) return null;
  if (!response.ok) {
    // Surfaced rather than swallowed. A page rendering "no notes yet" when the
    // API is actually broken is the same failure this whole project exists to
    // argue against -- a confident answer covering a gap.
    throw new Error(`GET ${path} failed: ${response.status} ${response.statusText}`);
  }

  // A 200 is not a promise of JSON. When the function behind this route fails
  // to boot, the platform answers with an HTML error page -- and parsing that
  // as JSON reports a syntax error at character 1, which says nothing about
  // what actually went wrong. Checking the type turns that into a message
  // naming the route and what it returned instead.
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    const body = (await response.text()).slice(0, 200);
    throw new Error(`GET ${path} returned ${contentType || "no content-type"}, not JSON: ${body}`);
  }
  return (await response.json()) as T;
}

export async function listNotes(): Promise<NoteSummary[]> {
  return (await read<NoteSummary[]>("/api/notes")) ?? [];
}

export async function getNote(ticker: string): Promise<NoteDetail | null> {
  return read<NoteDetail>(`/api/notes/${encodeURIComponent(ticker)}`);
}

/**
 * Display names for the five sections.
 *
 * Unknown keys fall back to the raw value rather than an empty heading, so a
 * section added on the Python side shows up looking unpolished instead of
 * vanishing from the page.
 */
const SECTION_TITLES: Record<string, string> = {
  business_overview: "Business overview",
  financial_health: "Financial health",
  risk_factors: "Risk factors",
  recent_developments: "Recent developments",
  valuation_context: "Valuation context",
};

export function sectionTitle(name: string): string {
  return SECTION_TITLES[name] ?? name;
}
