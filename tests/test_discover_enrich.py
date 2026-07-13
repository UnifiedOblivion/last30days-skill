"""U3 - enrichment stage: full pipeline pass per nominated topic.

The fault-tolerance contract is the point of these tests: one topic failing or
running past the batch budget must never sink the others, and the batch never
raises.
"""

import time
from unittest import mock

from lib import pipeline, schema


def _nomination(name: str, score: float = 50.0) -> pipeline.Nomination:
    return pipeline.Nomination(name=name, seed_score=score, items=[], summary=name)


def _report(topic: str) -> schema.Report:
    return schema.Report(
        topic=topic,
        range_from="2026-06-10",
        range_to="2026-07-10",
        generated_at="2026-07-10T00:00:00+00:00",
        provider_runtime=schema.ProviderRuntime(
            reasoning_provider="none",
            planner_model="deterministic",
            rerank_model="deterministic",
        ),
        query_plan=schema.QueryPlan(
            intent="factual",
            freshness_mode="balanced_recent",
            cluster_mode="none",
            raw_topic=topic,
            subqueries=[],
            source_weights={},
        ),
        clusters=[],
        ranked_candidates=[],
        items_by_source={},
        errors_by_source={},
    )


def test_enrich_all_success_preserves_order():
    nominations = [_nomination("Topic A"), _nomination("Topic B"), _nomination("Topic C")]

    def fake_run(*, topic, **_kwargs):
        return _report(topic)

    with mock.patch.object(pipeline, "run", side_effect=fake_run):
        enriched = pipeline.enrich_nominations(nominations, config={})

    assert [entry.nomination.name for entry in enriched] == ["Topic A", "Topic B", "Topic C"]
    assert all(entry.report is not None for entry in enriched)
    assert all(entry.error is None for entry in enriched)


def test_enrich_one_failure_does_not_sink_the_batch():
    nominations = [_nomination("Good"), _nomination("Bad"), _nomination("Also good")]

    def fake_run(*, topic, **_kwargs):
        if topic == "Bad":
            raise RuntimeError("upstream exploded")
        return _report(topic)

    with mock.patch.object(pipeline, "run", side_effect=fake_run):
        enriched = pipeline.enrich_nominations(nominations, config={})

    by_name = {entry.nomination.name: entry for entry in enriched}
    assert by_name["Good"].report is not None
    assert by_name["Also good"].report is not None
    assert by_name["Bad"].report is None
    assert "upstream exploded" in (by_name["Bad"].error or "")


def test_enrich_budget_expiry_drops_slow_topic_to_nomination_only():
    nominations = [_nomination("Fast"), _nomination("Slow")]

    def fake_run(*, topic, **_kwargs):
        if topic == "Slow":
            time.sleep(5)
        return _report(topic)

    with mock.patch.object(pipeline, "run", side_effect=fake_run):
        enriched = pipeline.enrich_nominations(
            nominations, config={}, budget_seconds=1.0, max_workers=2,
        )

    by_name = {entry.nomination.name: entry for entry in enriched}
    assert by_name["Fast"].report is not None
    assert by_name["Slow"].report is None
    assert "budget" in (by_name["Slow"].error or "")


def test_enrich_runs_as_internal_subrun():
    """Sub-runs must use the internal_subrun lane (no library context, capped
    inner workers) exactly like comparison-mode entity passes."""
    seen: dict[str, object] = {}

    def fake_run(*, topic, **kwargs):
        seen.update(kwargs)
        return _report(topic)

    with mock.patch.object(pipeline, "run", side_effect=fake_run):
        pipeline.enrich_nominations([_nomination("One")], config={})

    assert seen.get("internal_subrun") is True


def test_enrich_empty_nominations_returns_empty():
    assert pipeline.enrich_nominations([], config={}) == []
