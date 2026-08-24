import Link from "next/link";
import { notFound } from "next/navigation";
import { getNote, sectionTitle } from "@/lib/api";
import type { Claim, FilingLink, Section } from "@/lib/api";

type Params = { params: Promise<{ ticker: string }> };

// Same reason as the index: the data is in a database the build cannot reach.
export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: Params) {
  const { ticker } = await params;
  return { title: `${ticker.toUpperCase()} — equity-research-agent` };
}

/** Accession → SEC URL, as resolved by the API. Never rebuilt here. */
function linkIndex(filings: FilingLink[]): Map<string, string> {
  return new Map(filings.map((filing) => [filing.accession, filing.url]));
}

function Citations({ claim, links }: { claim: Claim; links: Map<string, string> }) {
  return (
    <span className="cites">
      {claim.chunks.map((chunk) => (
        <a
          key={`${chunk.accession}#${chunk.chunk_id}`}
          className="cite"
          href={links.get(chunk.accession) ?? "#"}
          target="_blank"
          rel="noreferrer"
          title={`Passage ${chunk.chunk_id} of filing ${chunk.accession}`}
        >
          {chunk.accession} #{chunk.chunk_id}
        </a>
      ))}
      {claim.facts.map((fact) => (
        <a
          key={`${fact.tag}-${fact.fiscal_period}`}
          className="cite"
          href={links.get(fact.accession) ?? "#"}
          target="_blank"
          rel="noreferrer"
          title={`XBRL tag ${fact.tag}, filed in ${fact.accession}`}
        >
          {/* The value, not just the tag. A citation that names a metric but
              hides its number is decorative -- the reader has to be able to
              check the figure against the filing without taking it on trust. */}
          {fact.tag} {fact.fiscal_period} = {fact.value.toLocaleString("en-US")}
        </a>
      ))}
    </span>
  );
}

function SectionBlock({ section, links }: { section: Section; links: Map<string, string> }) {
  return (
    <section>
      <h2>{sectionTitle(section.name)}</h2>
      {section.available ? (
        section.claims.map((claim, index) => (
          <p className="claim" key={index}>
            {claim.text}
            <Citations claim={claim} links={links} />
          </p>
        ))
      ) : (
        <p className="unavailable">
          Unavailable — {section.unavailable_reason ?? "no reason recorded"}.
        </p>
      )}
    </section>
  );
}

export default async function NotePage({ params }: Params) {
  const { ticker } = await params;
  const detail = await getNote(ticker);
  if (!detail) notFound();

  const { note } = detail;
  const links = linkIndex(detail.filings);

  return (
    <>
      <Link className="back" href="/">
        ← all companies
      </Link>

      <h1>{note.ticker}</h1>
      <p className="lede">
        CIK <span className="mono">{note.cik}</span>
        {detail.filings.length > 0 && (
          <>
            {" · sources: "}
            {detail.filings.map((filing, index) => (
              <span key={filing.accession}>
                {index > 0 && ", "}
                <a className="mono" href={filing.url} target="_blank" rel="noreferrer">
                  {filing.accession}
                </a>
              </span>
            ))}
          </>
        )}
      </p>

      {note.sections.map((section) => (
        <SectionBlock key={section.name} section={section} links={links} />
      ))}

      <h2>Coverage</h2>
      <p>
        Sections populated: {note.coverage.sections_available} of{" "}
        {note.coverage.sections_total}.
        {note.coverage.metrics_missing.length > 0 && (
          <>
            {" "}
            Metrics that could not be resolved from XBRL:{" "}
            {note.coverage.metrics_missing.join(", ")}.
          </>
        )}
      </p>
      <p className="mono">
        Run cost ${note.cost_usd.toFixed(4)} · {note.latency_seconds.toFixed(1)}s
      </p>

      <p className="disclaimer">
        This note summarises public filings. It is not investment advice, and
        nothing in it suggests any course of action.
      </p>
    </>
  );
}
