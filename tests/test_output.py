import src.output as out
from src.output import bound_output


def test_under_cap_unchanged():
    assert bound_output("hello", 4096) == "hello"


def test_over_cap_truncated_and_bounded():
    big = "x" * 10000
    out = bound_output(big, 4096)
    assert len(out.encode("utf-8")) <= 4096
    assert "truncated" in out


def test_skipped_sample_is_bounded_and_says_how_many_it_left_out():
    """The sample is capped at 20 because an unbounded list once wrote 3,000 entries /
    128 KB into a meta.json that list_metas() parses on every call. Capping silently
    would understate the damage, so the tail has to name the remainder."""
    from src.output import skipped_block

    class _Meta:
        file_count = 1
        skipped_counts = {"binary": 25}
        skipped = [f"binary: f{i}.png" for i in range(25)]

    out = skipped_block(_Meta())
    assert "25 readable file(s) were NOT loaded" in out
    assert out.count("binary: f") == 20, "the sample must stay bounded at 20"
    assert "and 5 more" in out, "the withheld count has to be visible"


# --- the cap must measure what the CLIENT measures --------------------------------
def test_the_cap_counts_json_encoded_size_not_raw_bytes():
    """The observed failure: a reply trimmed to exactly 131,072 raw bytes arrived at the
    client as 134,245 characters and was REFUSED, so a finished 30-call batch surfaced as
    an error with its result dumped to a temp file. MCP results travel as
    {"result": "..."} and every newline costs two characters — markdown findings are
    newline-dense, so raw bytes systematically under-count by a few percent.
    """
    cap = 131_072
    dense = ("### chunk finding\n" + "x" * 60 + "\n") * 3000
    assert out.encoded_len(dense) > len(dense.encode()), "escaping should inflate this"

    bounded = out.bound_output(dense, cap)
    assert out.encoded_len(bounded) <= cap, "still over the limit the client enforces"


def test_bounding_is_a_no_op_when_the_encoded_form_fits():
    text = "a plain short answer\nwith two lines"
    assert out.bound_output(text, 131_072) == text


def test_a_quote_heavy_payload_is_also_brought_under_the_cap():
    """Quotes and backslashes escape too, so a JSON-ish payload inflates harder than
    prose. The shrink loop must converge on content-dependent ratios, not a fixed margin.
    """
    cap = 4096
    payload = ('{"k": "v", "path": "C:\\Users\\x"}\n' * 400)
    bounded = out.bound_output(payload, cap)
    assert out.encoded_len(bounded) <= cap
    assert "truncated" in bounded


def _body(bounded: str) -> str:
    """Everything before the truncation notice — the part the caller actually asked for."""
    return bounded.split("\n…[truncated")[0]


def test_escape_heavy_content_still_comes_back_with_content_in_it():
    """Staying under the cap is half the job; the other half is not returning an empty
    reply. Correcting by the OVERSHOOT assumes one encoded character per raw byte, so
    content that at least doubles under encoding drove the kept length to zero and handed
    back the truncation notice alone. A control character costs six characters, which is
    enough on its own -- reachable through rlm_read_chunk or rlm_exec output.
    """
    cap = 4096
    for name, payload in (
        ("control chars", "\x01\x02\x03abc" * 20_000),
        ("all quotes", '"' * 100_000),
        ("all newlines", "\n" * 100_000),
    ):
        bounded = out.bound_output(payload, cap)
        assert out.encoded_len(bounded) <= cap, name
        assert len(_body(bounded)) > cap // 8, (
            f"{name}: returned {len(_body(bounded))} characters of content out of a "
            f"{cap} cap — effectively an empty reply")


def test_prose_still_fills_most_of_the_cap():
    """The proportional step must not overcorrect the ordinary case it never broke."""
    bounded = out.bound_output("The quick brown fox jumps over the lazy dog. " * 2500, 4096)

    assert out.encoded_len(bounded) <= 4096
    assert len(_body(bounded)) > 3500, "prose lost most of its budget to the correction"
