import Link from "next/link";
import { listNotes } from "@/lib/api";

export const dynamic = "force-dynamic";

const REPO = "https://github.com/berkaykoklu/equity-research-agent";
const HOME = "https://berkaykoklu.vercel.app";

function generatedOn(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

export default async function Home() {
  const notes = await listNotes();

  return (
    <main className="relative z-10 mx-auto w-full max-w-[64rem] px-6 py-16 sm:py-24">
      <a href={HOME} className="font-mono text-[0.76rem] text-low transition-colors hover:text-mid">
        ← berkaykoklu.vercel.app
      </a>

      <h1 className="display mt-6 text-[clamp(2.2rem,6vw,3.8rem)]">
        Research notes that cannot cite<br />what does not exist
      </h1>
      <p className="mt-6 max-w-[62ch] text-[1.05rem] leading-relaxed text-mid">
        Language models are fluent about company financials and confidently
        wrong about the numbers. Every sentence below carries the filing passage
        it came from, and every figure is read from the company&rsquo;s own filed
        data rather than written by the model. Click any citation to land on the
        filing at SEC.
      </p>

      <section className="mt-14 sm:mt-20">
        <h2 className="label mb-4">NOTES</h2>
        {notes.length === 0 ? (
          <div className="lift rounded-[14px] p-6">
            <p className="text-mid">
              No notes are stored yet. They are generated deliberately from the
              command line, never on demand — producing one costs real money, so
              an endpoint that could generate is an endpoint a stranger can make
              expensive.
            </p>
          </div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {notes.map((note) => (
              <Link
                key={note.ticker}
                href={`/notes/${note.ticker}`}
                className="lift group rounded-[14px] p-5 transition-shadow duration-200 hover:shadow-[0_0_0_1px_var(--color-brand-deep),0_20px_50px_-26px_rgba(0,0,0,.95)]"
              >
                <p className="display text-[1.9rem] text-brand">{note.ticker}</p>
                <p className="mt-2 font-mono text-[0.72rem] text-low tnum">
                  CIK {note.cik}
                </p>
                <p className="mt-3 border-t border-line pt-3 text-[0.82rem] text-mid tnum">
                  {note.sections_available} of {note.sections_total} sections
                  <span className="text-low"> · {generatedOn(note.generated_at)}</span>
                </p>
              </Link>
            ))}
          </div>
        )}
      </section>

      <section className="mt-16 sm:mt-24">
        <h2 className="display text-[clamp(1.5rem,3.6vw,2.1rem)]">How it works</h2>
        <div className="mt-7 grid gap-4 sm:grid-cols-2">
          {[
            {
              t: "Words and numbers travel separately",
              d: "The model is never able to type a figure. It names a metric it wants; code looks that name up in the company's XBRL data and attaches the real tag, period and value. A model that cannot write a number cannot misquote one.",
            },
            {
              t: "The checker is code, not a model",
              d: "Whether a citation resolves and whether a figure matches the filing are decidable questions, so they are decided. The same function runs inside the graph and again as the CI gate every change must pass.",
            },
            {
              t: "Gaps are reported, not guessed",
              d: "A chapter whose heading cannot be found is reported missing with the reason. A claim failing verification twice is dropped, not shipped — a visible gap beats a confident guess.",
            },
            {
              t: "Nothing here generates",
              d: "Producing a note costs real money and takes about eighty seconds, so it happens deliberately from the CLI and the result is stored. This site only reads.",
            },
          ].map((c) => (
            <div key={c.t} className="lift h-full rounded-[14px] p-5">
              <h3 className="text-[1rem] font-semibold">{c.t}</h3>
              <p className="mt-2 text-[0.89rem] leading-relaxed text-mid">{c.d}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="mt-16 border-t border-line pt-8 text-[0.84rem] text-low sm:mt-24">
        <p className="mb-3 italic">
          These notes summarise public filings. They are not investment advice,
          and nothing in them suggests any course of action.
        </p>
        <a href={REPO} className="transition-colors hover:text-mid">
          Code and every figure — github.com/berkaykoklu/equity-research-agent
        </a>
      </footer>
    </main>
  );
}
