from src.output import bound_output


def test_under_cap_unchanged():
    assert bound_output("hello", 4096) == "hello"


def test_over_cap_truncated_and_bounded():
    big = "x" * 10000
    out = bound_output(big, 4096)
    assert len(out.encode("utf-8")) <= 4096
    assert "truncated" in out
