"""Bounded manual imports and descriptive sentiment totals; no file persistence."""

import csv
import io

MAX_IMPORTED_COMMENTS = 100
MAX_CSV_BYTES = 256 * 1024
MAX_CHARACTERS = 2000
SENTIMENT_LABELS = ("positive", "neutral", "negative")


def validate_comments(comments, limit=MAX_IMPORTED_COMMENTS):
    if not isinstance(comments, list) or not comments:
        raise ValueError("Enter at least one nonblank comment.")
    if len(comments) > limit:
        raise ValueError(f"Analyze at most {limit} comments at a time.")
    cleaned = []
    for index, comment in enumerate(comments):
        if not isinstance(comment, str) or not comment.strip():
            raise ValueError(f"Comment {index + 1} must contain nonblank text.")
        if len(comment) > MAX_CHARACTERS:
            raise ValueError(f"Comment {index + 1} must contain at most 2,000 characters.")
        if "\x00" in comment:
            raise ValueError(f"Comment {index + 1} contains an unsupported null character.")
        cleaned.append(comment.strip())
    return cleaned


def parse_pasted_comments(text, limit=MAX_IMPORTED_COMMENTS):
    if not isinstance(text, str):
        raise ValueError("Paste comment text, one comment per line.")
    # Bound total work before splitting a pasted post export.
    if len(text) > MAX_CSV_BYTES:
        raise ValueError("Pasted comments are too large. Use at most 256 KiB of text.")
    return validate_comments([line for line in text.splitlines() if line.strip()], limit)


def parse_comment_csv(data, limit=MAX_IMPORTED_COMMENTS):
    if not isinstance(data, bytes) or not data or len(data) > MAX_CSV_BYTES:
        raise ValueError("Upload a nonempty CSV file of at most 256 KiB.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("CSV files must use UTF-8 text encoding.") from None
    if "\x00" in text:
        raise ValueError("The CSV contains unsupported null characters.")
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        headers = reader.fieldnames or []
        candidates = [name for name in headers if name in {"text", "comment", "comment_text"}]
        if len(headers) != len(set(headers)) or len(candidates) != 1:
            raise ValueError("CSV needs exactly one comment column named text, comment, or comment_text.")
        column = candidates[0]
        comments = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("The CSV contains malformed rows.")
            if row[column].strip():
                comments.append(row[column])
                if len(comments) > limit:
                    raise ValueError(f"Analyze at most {limit} comments at a time.")
    except csv.Error:
        raise ValueError("The CSV is malformed. Check its header, quoting, and rows.") from None
    return validate_comments(comments, limit)


def aggregate_sentiment(results):
    if not results:
        raise ValueError("There are no analyzed comments to summarize.")
    counts = dict.fromkeys(SENTIMENT_LABELS, 0)
    for result in results:
        label = result.get("label")
        if label not in counts:
            raise ValueError("The API returned an invalid sentiment label.")
        counts[label] += 1
    return {
        "total": len(results),
        "counts": counts,
        "percentages": {label: round(count / len(results) * 100, 2) for label, count in counts.items()},
    }
