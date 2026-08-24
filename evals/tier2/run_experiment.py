"""Tier 2: judged quality, tracked over time.

Tier 1 asks whether a note is *verifiable* -- every citation resolves, every
figure matches, nothing reads as advice. Deterministic code can answer that,
so it gates every merge.

This asks whether a note is any *good*: is it grounded in its sources, does it
answer the question, is the retrieved context actually relevant. Only a model
can judge that, which makes it slow, nondeterministic and metered.

**This never gates a merge.** A gate that fails randomly is one people learn to
ignore, and an ignored gate is worse than none because it still looks like
protection. It runs after merge and reports a trend.

Run it deliberately:

    uv run python evals/tier2/run_experiment.py

Requires OPIK_API_KEY, OPIK_WORKSPACE and the tickers to be ingested already.
"""

import os
import sys

from era.config import Settings
from era.edgar.client import EdgarClient
from era.edgar.filings import latest_filings, resolve_cik
from era.edgar.xbrl import normalize_facts
from era.graph.build import build_research_graph
from era.graph.models import CHEAP_MODEL, DRAFTING_MODEL, build_model
from era.index.embeddings import VoyageEmbedder
from era.index.store import PgVectorStore
from era.report.assemble import render_markdown

# The judge is the cheap model on purpose. Judging is high-volume and lower
# stakes than drafting, and this project has a hard total budget -- see the
# unit-economics section of the design doc.
JUDGE_MODEL = f"openai/{CHEAP_MODEL}"

# Three tickers, not ten. Each full run costs roughly $0.10 to draft plus judge
# calls, and every ticker must first be ingested -- which on Voyage's free tier
# takes ~20 minutes for a large filing. Three companies of genuinely different
# shape says more about robustness than ten of the same shape would, and the
# design doc's $7 ceiling makes the difference matter.
GOLDEN_TICKERS = ("AAPL", "KO", "JNJ")

DATASET_NAME = "era-golden-tickers"


def _research(ticker: str) -> tuple[str, str]:
    """Produce one note and the source text behind it.

    Returns (rendered note, concatenated cited chunk text). The judge needs the
    sources to score grounding -- handing it only the note would let it grade
    fluency and call that faithfulness.
    """
    settings = Settings()  # type: ignore[call-arg]
    from pathlib import Path

    with (
        EdgarClient(
            user_agent=settings.edgar_user_agent, cache_dir=Path(".cache/edgar-live")
        ) as client,
        PgVectorStore(dsn=settings.database_url) as store,
    ):
        cik = resolve_cik(client, ticker)
        annual = next(f for f in latest_filings(client, cik) if f.form == "10-K")
        facts = normalize_facts(
            client.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
        )
        graph = build_research_graph(
            build_model(model_name=DRAFTING_MODEL, api_key=settings.openai_api_key),
            store,
            VoyageEmbedder(api_key=settings.voyage_api_key),
            facts,
        )
        result = graph.invoke({"ticker": ticker, "cik": cik, "accession": annual.accession})
        note = result["note"]

        sources: list[str] = []
        for section in note.sections:
            for claim in section.claims:
                for ref in claim.chunks:
                    stored = store.get(ref.accession, ref.chunk_id)
                    if stored is not None:
                        sources.append(stored.text)

    # Deduplicate while preserving order: several claims cite the same passage,
    # and repeating it inflates the context the judge sees without adding
    # anything for it to check against.
    seen: set[str] = set()
    unique = [text for text in sources if not (text in seen or seen.add(text))]
    return render_markdown(note), "\n\n".join(unique)


def task(item: dict[str, str]) -> dict[str, str]:
    ticker = item["ticker"]
    note, context = _research(ticker)
    return {
        "input": f"Write a cited research note on {ticker} from its latest 10-K.",
        "output": note,
        "context": context,
    }


def _export_credentials() -> None:
    """Bridge .env into the process environment for SDKs that read it directly.

    This project loads credentials through `Settings`, but Opik and the OpenAI
    client both read `os.environ`. Without this, a key sitting correctly in
    `.env` produces an authentication error from deep inside a library -- a
    configuration problem wearing someone else's error message. Existing
    environment variables win, so CI (which sets real ones) is unaffected.
    """
    settings = Settings()  # type: ignore[call-arg]
    for name, value in (
        ("OPIK_API_KEY", settings.opik_api_key),
        ("OPIK_WORKSPACE", settings.opik_workspace),
        ("OPENAI_API_KEY", settings.openai_api_key),
    ):
        if value and not os.environ.get(name):
            os.environ[name] = value


def main() -> int:
    _export_credentials()

    import opik
    from opik.evaluation import evaluate
    from opik.evaluation.metrics import AnswerRelevance, ContextPrecision, Hallucination

    client = opik.Opik()
    dataset = client.get_or_create_dataset(name=DATASET_NAME)
    dataset.insert([{"ticker": ticker} for ticker in GOLDEN_TICKERS])

    evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=[
            Hallucination(model=JUDGE_MODEL),
            AnswerRelevance(model=JUDGE_MODEL),
            ContextPrecision(model=JUDGE_MODEL),
        ],
        experiment_name=os.environ.get("GITHUB_SHA", "local")[:12],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
