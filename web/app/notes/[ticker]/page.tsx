import Link from "next/link";
import { notFound } from "next/navigation";
import { getNote, sectionTitle } from "@/lib/api";
import type { Claim, FilingLink, Section } from "@/lib/api";

type Params = { params: Promise<{ ticker: string }> };

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: Params) {
  const { ticker } = await params;
  return { title: `${ticker.toUpperCase()} — research note` };
}

/** Accession to SEC URL, as resolved by the API. Never rebuilt here: the rule
 *  uses two different CIK spellings and fails silently when they are mixed up,
 *  so it lives next to the code that knows EDGAR's layout. */
function linkIndex(filings: FilingLink[]): Map<string, string> {
  return new Map(filings.map((filing) => [filing.accession, filing.url]));
}

function Citations({ claim, links }: { claim: Claim; links: Map<string, string> }) {
  const chip =
    "inline-flex items-center rounded border border-line bg-raised px-1.5 py-0.5 font-mono text-[0.68rem] text-low transition-colors hover:border-brand-deep hover:text-brand";

  return (
    <span className="mt-1.5 flex flex-wrap gap-1">
      {claim.chunks.map((chunk) => (
        <a
          key={`${chunk.accession}#${chunk.chunk_id}`}
          className={chip}
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
          className={chip}
          href={links.get(fact.accession) ?? "#"}
          target="_blank"
          rel="noreferrer"
          title={`XBRL tag ${fact.tag}, filed in ${fact.accession}`}
        >
          {/* The value, not just the tag. A citation naming a metric but hiding
              its number is decorative — the reader has to be able to check it. */}
          {fact.tag} {fact.fiscal_period} = {fact.value.toLocaleString("en-US")}
        </a>
      ))}
    </span>
  );
}

function SectionBlock({ section, links }: { section: Section; links: Map<string, string> }) {
  return (
    <section className="mt-12">
      <h2 className="display text-[clamp(1.4rem,3.4vw,2rem)]">{sectionTitle(section.name)}</h2>
      {section.available ? (
        <div className="mt-5 space-y-5">
          {section.claims.map((claim, index) => (
            <p key={index} className="max-w-[68ch] text-[0.98rem] leading-relaxed text-mid">
              {claim.text}
              <Citations claim={claim} links={links} />
            </p>
          ))}
        </div>
      ) : (
        <p className="mt-5 rounded-[14px] border border-block/35 bg-block/[0.06] p-4 text-[0.92rem] text-hi">
          Unavailable — {section.unavailable_reason ?? "no reason recorded"}. Reported
          rather than guessed.
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
    <main className="relative z-10 mx-auto w-full max-w-[64rem] px-6 py-16 sm:py-24">
      <Link href="/" className="font-mono text-[0.76rem] text-low transition-colors hover:text-mid">
        ← all companies
      </Link>

      <h1 className="display mt-6 text-[clamp(2.6rem,8vw,4.5rem)]">{note.ticker}</h1>
      <p className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[0.78rem] text-low">
        <span>CIK {note.cik}</span>
        {detail.filings.map((filing) => (
          <a key={filing.accession} href={filing.url} target="_blank" rel="noreferrer"
             className="text-brand transition-colors hover:text-brand-lit">
            {filing.accession}
          </a>
        ))}
      </p>

      {note.sections.map((section) => (
        <SectionBlock key={section.name} section={section} links={links} />
      ))}

      <section className="mt-14 border-t border-line pt-8">
        <h2 className="label mb-4">COVERAGE</h2>
        <div className="flex flex-wrap gap-8">
          <div>
            <p className="display text-[2rem] tnum">
              {note.coverage.sections_available}
              <span className="text-low">/{note.coverage.sections_total}</span>
            </p>
            <p className="label mt-1">SECTIONS POPULATED</p>
          </div>
          <div>
            <p className="display text-[2rem] tnum">${note.cost_usd.toFixed(4)}</p>
            <p className="label mt-1">COST TO PRODUCE</p>
          </div>
          <div>
            <p className="display text-[2rem] tnum">{note.latency_seconds.toFixed(0)}s</p>
            <p className="label mt-1">TIME TO PRODUCE</p>
          </div>
        </div>
        {note.coverage.metrics_missing.length > 0 && (
          <p className="mt-5 text-[0.88rem] text-mid">
            Metrics that could not be resolved from XBRL:{" "}
            <span className="font-mono text-hi">{note.coverage.metrics_missing.join(", ")}</span>.
          </p>
        )}
      </section>

      <footer className="mt-12 border-t border-line pt-8 text-[0.84rem] italic text-low">
        This note summarises public filings. It is not investment advice, and
        nothing in it suggests any course of action.
      </footer>
    </main>
  );
}
