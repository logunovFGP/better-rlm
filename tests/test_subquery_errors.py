"""sub_query surfaces a transport failure as a SubResult, never as an exception.

The batch path has its own error accounting (test_batch_failfast); this pins the
single-query sibling, which rlm_sub_query and the reduce pass both go through. A raised
exception here would abort a whole tool call instead of reporting one failed answer.
"""

import src.subquery as sq


def test_sub_query_reports_a_transport_failure_as_an_error_result(monkeypatch, cfg):
    def boom(cfg, model, prompt, max_tokens, system):
        raise RuntimeError("socket closed")

    monkeypatch.setattr(sq, "_call", boom)
    res = sq.sub_query(cfg, "summarize", "m")
    assert res.error == "socket closed"
    assert (res.answer, res.input_tokens, res.output_tokens) == ("", 0, 0)


def test_sub_query_returns_the_model_the_transport_actually_used(monkeypatch, cfg):
    """On OAuth, models.select maps a configured id to its closest subscription sibling,
    so the only way to know what ran is to read it back off the response."""
    monkeypatch.setattr(sq, "_call",
                        lambda cfg, model, prompt, max_tokens, system: ("hi", 3, 1, "haiku-actual"))
    res = sq.sub_query(cfg, "summarize", "requested-model")
    assert (res.answer, res.model, res.error) == ("hi", "haiku-actual", None)
