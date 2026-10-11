import pytest

from src.comment_import import aggregate_sentiment, parse_comment_csv, parse_pasted_comments


def test_manual_import_preserves_independent_comments_and_duplicates():
    assert parse_pasted_comments(" First\n\nSecond\nFirst ") == ["First", "Second", "First"]


def test_csv_quotes_multiline_text_and_bom_are_preserved():
    assert parse_comment_csv('\ufefftext,id\n"Hello, friend",1\n"A comment\nwith two lines",2\n'.encode()) == [
        "Hello, friend", "A comment\nwith two lines",
    ]


@pytest.mark.parametrize("data", [b"", b"x" * (256 * 1024 + 1), b"\xff", b"text\n\x00\n", b"wrong\nhello\n", b"text,comment\na,b\n", b"text,text\na,b\n", b'text\n"unterminated', b"text,id\na,1,extra\n", b"text,id\na\n", b"text\n\n"], ids=lambda _: "invalid_csv")
def test_invalid_csv_is_rejected(data):
    with pytest.raises(ValueError):
        parse_comment_csv(data)


@pytest.mark.parametrize("text", [" ", "x" * 2001, "\n".join(["comment"] * 101), "x" * (256 * 1024 + 1)], ids=lambda _: "invalid_text")
def test_manual_import_bounds(text):
    with pytest.raises(ValueError):
        parse_pasted_comments(text)


def test_exact_limits_are_accepted():
    assert len(parse_pasted_comments("\n".join(["x" * 2000] * 100))) == 100


def test_aggregate_counts_predictions_rather_than_model_scores():
    results = [{"label": label, "scores": {"positive": 0.99}} for label in ["positive", "neutral", "negative", "negative"]]
    assert aggregate_sentiment(results) == {
        "total": 4, "counts": {"positive": 1, "neutral": 1, "negative": 2},
        "percentages": {"positive": 25.0, "neutral": 25.0, "negative": 50.0},
    }


@pytest.mark.parametrize("results", [[], [{"label": "no_detected_signals"}], [{}]])
def test_no_fake_neutral_or_empty_percentages(results):
    with pytest.raises(ValueError):
        aggregate_sentiment(results)
