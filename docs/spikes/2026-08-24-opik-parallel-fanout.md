# Spike: does Opik trace the graph's parallel fan-out?

**Date:** 2026-08-24
**Question:** the graph researches five sections concurrently. Do those nodes
appear as named spans under a single trace, or does the fan-out fragment into
several traces — or vanish?

This blocked committing to Opik as the tracing layer. Opik's documentation does
not cover LangGraph fan-out, and there is a known async context-propagation
caveat, so the answer had to be measured rather than assumed.

## Method

Ran the real `build_research_graph` with a scripted fake model — the question is
about tracing, not generation, so a fake isolates it and costs nothing. Attached
`OpikTracer` with `graph=graph.get_graph(xray=True)`, then read the spans back
**from Opik's own API** rather than trusting the local tracer object.

## Result

    sections: 5
    traces created: 1
    spans on trace: 8
    names: ['business_overview', 'finalise', 'financial_health',
            'recent_developments', 'risk_factors', 'should_retry',
            'valuation_context', 'verify']

One trace. All five concurrent section nodes present by name, plus `verify`,
`should_retry` and `finalise`.

## Decision

**Proceed with synchronous fan-out and `OpikTracer` as written.** No fallback,
no manual span propagation, no restructuring.

Tracing is wired into `era research` through `measured(...)`, and is
deliberately best-effort: if Opik is unconfigured or unreachable, the tracer is
skipped and the run still records cost and latency. Telemetry must not be able
to fail a run that is otherwise fine.
