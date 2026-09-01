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
