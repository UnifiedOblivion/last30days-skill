"""Per-call cost markers for real-cost search billing.

The research engine runs as a subprocess of the last30days-com research-worker,
which captures this engine's stderr and parses ``[cost]`` markers back out to
settle each run on its real provider spend (worker side: research-worker/cost.py
``parse_run_usage``). This module emits markers in the EXACT grammar that parser
expects:

    [cost] provider=<slug> model=<id> prompt_tokens=<n> completion_tokens=<n> \\
           reasoning_tokens=<n> cached_tokens=<n> calls=<n> cost_usd=<float>

The engine's providers do not expose token counts and the dominant spend is
per-call paid APIs (ScrapeCreators credits, web-search backends, Perplexity), so
cost is modeled PER CALL from a flat rate card rather than token-metered. Token
fields stay 0; ``cost_usd`` carries the rate-card price for the call.

Reasoning-LLM entries are token-derived *estimates* (planner/rerank/fun JSON
calls) because the clients do not surface usage metadata; they are near-zero vs
paid search/social APIs but included for completeness.

Rate card validated 2026-06-17 against vendor pricing pages (see inline sources).
Cost accounting must NEVER break a research run: every function here is
exception-safe and returns/does-nothing on bad input.

Plan: docs/plans/2026-06-17-001-feat-measured-search-cost-markers-plan.md (U1).
"""

from __future__ import annotations

import sys
from typing import Optional, TextIO

# ---------------------------------------------------------------------------
# Rate card: USD per successful call. Paid APIs use published per-request /
# per-credit list prices for the endpoints this engine actually calls.
# Unknown providers price at 0 (a missing price never fabricates cost).
# ---------------------------------------------------------------------------
RATE_CARD: dict[str, float] = {
    # ScrapeCreators: 1 credit per request. Freelance pack = $47 / 25,000 credits
    # ($1.88/1k) — scrapecreators.com pricing, 2026-06-17.
    "scrapecreators": 0.00188,
    # Brave Search API: $5.00 / 1,000 web-search requests — Brave API pricing.
    "brave": 0.005,
    # Exa: $7.00 / 1,000 search requests with contents (10 results + text).
    # grounding.exa_search requests contents.text — exa.ai/pricing, 2026-06-17.
    "exa": 0.007,
    # Serper: $1.00 / 1,000 queries on the $50 starter pack — serper.dev.
    # (Scale tier is $0.50/1k; starter is the conservative default.)
    "serper": 0.001,
    # Parallel Search API: $5.00 / 1,000 requests (10 results + excerpts).
    # docs.parallel.ai/getting-started/pricing, 2026-06-17.
    "parallel": 0.005,
    # Reasoning LLMs — flat per-call estimates (no token data exposed).
    "gemini": 0.0005,
    "openai": 0.0005,
    "xai": 0.0005,
    "openrouter": 0.0010,
}

# Model-keyed overrides (provider:model) for cases where price varies by model.
RATE_CARD_BY_MODEL: dict[str, float] = {
    # Perplexity Sonar Pro via OpenRouter: low-context request fee $6/1k ($0.006)
    # plus typical short-synthesis token cost — docs.perplexity.ai/guides/pricing.
    # Rounded to $0.010/call as a flat settlement estimate.
    "perplexity:sonar-pro": 0.010,
    # Sonar Deep Research: Perplexity published examples run $0.41 (low) to
    # $1.32 (high); $1.00/call is a mid-range flat estimate for settlement.
    "perplexity:sonar-deep-research": 1.00,
}


def price_for(provider: str, model: str = "") -> float:
    """Resolve the per-call USD price for a provider/model from the rate card.
    Unknown provider/model -> 0.0 (never fabricate cost)."""
    try:
        if model:
            keyed = RATE_CARD_BY_MODEL.get(f"{provider}:{model}")
            if keyed is not None:
                return float(keyed)
        return float(RATE_CARD.get(provider, 0.0))
    except Exception:  # pragma: no cover - defensive
        return 0.0


def emit_cost(
    provider: str,
    *,
    model: str = "",
    calls: int = 1,
    cost_usd: Optional[float] = None,
    stream: Optional[TextIO] = None,
) -> float:
    """Write one ``[cost]`` marker for a billable call and return the cost.

    ``cost_usd`` defaults to the rate-card price for (provider, model) times
    ``calls``. Pass an explicit ``cost_usd`` only when a call site already knows
    the real cost. A ``model`` is included so the worker can build a per-model
    breakdown; for paid APIs with no model, pass the endpoint/source as model
    (e.g. ``model="reddit"``) or leave it blank (cost still counts toward the
    run total, just not the per-model rows).

    Never raises.
    """
    out = stream if stream is not None else sys.stderr
    try:
        n = int(calls) if calls else 0
        if cost_usd is None:
            cost = price_for(provider, model) * max(n, 0)
        else:
            cost = float(cost_usd)
        out.write(
            f"[cost] provider={provider} model={model} "
            f"prompt_tokens=0 completion_tokens=0 "
            f"reasoning_tokens=0 cached_tokens=0 "
            f"calls={n} cost_usd={cost:.6f}\n"
        )
        try:
            out.flush()
        except Exception:  # pragma: no cover - defensive
            pass
        return cost
    except Exception:  # pragma: no cover - defensive
        return 0.0
