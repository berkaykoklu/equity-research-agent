import Link from "next/link";
import { listNotes } from "@/lib/api";

export const metadata = {
  title: "equity-research-agent",
};

// Rendered per request, not at build. The notes live in Postgres, which is not
// reachable while the build runs -- prerendering would bake in "no notes yet"
// and serve that until the next deploy. The fetch layer still caches, so this
// costs a query per five minutes, not per visitor.
export const dynamic = "force-dynamic";

function generatedOn(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

export default async function Home() {
  const notes = await listNotes();

  return (
    <>
      <h1>Research notes from SEC filings</h1>
      <p className="lede">
        Every sentence below is traceable to the paragraph it came from, and every
        figure comes from the company&rsquo;s own filed data rather than from a
        language model. Click any citation to land on the filing itself.
      </p>

      {notes.length === 0 ? (
        <p className="unavailable">
          No notes are stored yet. They are generated deliberately from the
          command line, not on demand.
        </p>
      ) : (
        <ul className="companies">
          {notes.map((note) => (
            <li key={note.ticker}>
              <Link href={`/notes/${note.ticker}`}>
                <span className="ticker">{note.ticker}</span>
                <span className="meta">
                  {note.sections_available} of {note.sections_total} sections
                  {" · "}
                  {generatedOn(note.generated_at)}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}

      <div className="panel">
        <h3>How this works</h3>
        <ul>
          <li>
            <strong>Words and numbers travel separately.</strong> The model is
            never able to type a figure. It names a metric it wants; code looks
            that name up in the company&rsquo;s XBRL data and attaches the real
            tag, period and value.
          </li>
          <li>
            <strong>The checker is code, not another model.</strong> Whether a
            citation resolves and whether a figure matches the filing are
            decidable questions, so they are decided rather than judged.
          </li>
          <li>
            <strong>Gaps are reported, not guessed.</strong> A chapter that
            cannot be located, or a claim that fails verification twice, is
            dropped and said so — never smoothed over.
          </li>
          <li>
            <strong>Nothing here generates.</strong> These notes were produced
            in advance and stored. The site only reads.
          </li>
        </ul>
      </div>

      <p className="disclaimer">
        These notes summarise public filings. They are not investment advice, and
        nothing in them suggests any course of action.
      </p>
    </>
  );
}
