"""YouTube connector checks are offline HTTP mocks, not live API verification."""

import json

import pytest
import requests

from src import platform_comments as platform

VIDEO_ID = "abcDEF123_-"
SECRET_FIXTURE = "test-only-do-not-echo-key"


class Response:
    def __init__(self, payload=None, status=200, body=None, headers=None):
        self.status_code = status
        self.body = body if body is not None else json.dumps(payload).encode("utf-8")
        self.headers = headers or {}
        self.closed = False
        self.reads = 0

    def iter_content(self, chunk_size):
        self.reads += 1
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset:offset + chunk_size]

    def close(self):
        self.closed = True


def item(comment):
    return {"snippet": {"topLevelComment": {"snippet": {"textDisplay": comment}}, "totalReplyCount": 10},
            "replies": {"comments": [{"snippet": {"textDisplay": "A reply never retrieved"}}]}}


def mock_get(monkeypatch, response):
    calls = []

    def get(url, **options):
        calls.append((url, options))
        return response

    monkeypatch.setattr(platform.requests, "get", get)
    return calls


@pytest.mark.parametrize("url", [
    f"https://youtube.com/watch?v={VIDEO_ID}",
    f"https://www.youtube.com/watch?v={VIDEO_ID}&t=10&feature=shared",
    f"https://m.youtube.com/watch?v={VIDEO_ID}",
    f"https://youtu.be/{VIDEO_ID}?si=ignored",
    f"https://www.youtube.com/shorts/{VIDEO_ID}",
    f"https://www.youtube.com/embed/{VIDEO_ID}/",
    f"  https://www.youtube.com/watch?v={VIDEO_ID}  ",
])
def test_supported_urls_extract_only_video_id(url):
    assert platform.parse_youtube_video_id(url) == VIDEO_ID


@pytest.mark.parametrize("url", [
    None, 123, "", VIDEO_ID, "http://youtube.com/watch?v=" + VIDEO_ID,
    "https://youtube.com.evil.test/watch?v=" + VIDEO_ID,
    "https://evil.test/youtube.com/watch?v=" + VIDEO_ID,
    "https://user:pass@youtube.com/watch?v=" + VIDEO_ID,
    "https://youtube.com@127.0.0.1/watch?v=" + VIDEO_ID,
    "https://youtube.com:443/watch?v=" + VIDEO_ID,
    "https://youtube.com:bad/watch?v=" + VIDEO_ID,
    "https://youtube.com./watch?v=" + VIDEO_ID,
    "https://127.0.0.1/watch?v=" + VIDEO_ID,
    "file:///watch?v=" + VIDEO_ID,
    "https://youtube.com/watch", "https://youtube.com/watch?v=short",
    "https://youtube.com/watch?v=" + VIDEO_ID + "&v=" + VIDEO_ID,
    "https://youtu.be/" + VIDEO_ID + "/extra",
    "https://youtube.com/redirect?q=https://example.com",
    "https://youtube.com/shorts/../watch?v=" + VIDEO_ID,
    "https://youtube.com/watch?v=" + VIDEO_ID + "\n",
    "https://youtube.com/wa\ntch?v=" + VIDEO_ID,
    "https://youtube.com/watch?v=" + VIDEO_ID + "\x00",
    "https://youtube.com/watch?v=" + VIDEO_ID + "&" + "a=1&" * 21,
    "https://youtube.com/watch?v=" + "x" * 2048,
])
def test_untrusted_or_malformed_urls_never_reach_http(monkeypatch, url):
    calls = mock_get(monkeypatch, Response({"items": []}))
    with pytest.raises(platform.InvalidPostURLError):
        platform.fetch_youtube_comments(url, api_key=SECRET_FIXTURE)
    assert calls == []


@pytest.mark.parametrize("limit", [0, 101, True, 1.5, "100", None])
def test_limit_is_strict_and_bounded_before_http(monkeypatch, limit):
    calls = mock_get(monkeypatch, Response({"items": []}))
    with pytest.raises(ValueError, match="integer between 1 and 100"):
        platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}", limit=limit, api_key=SECRET_FIXTURE)
    assert calls == []


def test_missing_key_has_clear_optional_setup_and_no_request(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    calls = mock_get(monkeypatch, Response({"items": []}))
    with pytest.raises(platform.MissingPlatformAccessError, match="YOUTUBE_API_KEY.*paste/upload"):
        platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}")
    assert calls == []


def test_single_static_request_plain_top_level_text_and_no_pagination(monkeypatch):
    response = Response({"items": [item("  thanks!  "), item("bad result")], "nextPageToken": "private-upstream-token"})
    calls = mock_get(monkeypatch, response)
    result = platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}?si=ignored", limit=2, api_key=SECRET_FIXTURE)
    assert len(calls) == 1
    url, options = calls[0]
    assert url == platform.YOUTUBE_ENDPOINT
    assert options["params"] == {"part": "snippet", "videoId": VIDEO_ID, "maxResults": 2,
                                 "order": "time", "textFormat": "plainText", "key": SECRET_FIXTURE}
    assert options["timeout"] == (5, 20)
    assert options["allow_redirects"] is False and options["stream"] is True
    assert result["comments"] == ["  thanks!  ", "bad result"]
    assert result["has_more"] is True
    assert result["scope"] == "retrieved_top_level_comments"
    assert result["retrieved_count"] == result["received_count"] == 2
    assert SECRET_FIXTURE not in str(result) and "private-upstream-token" not in str(result)
    assert response.closed


def test_blank_and_oversized_comments_are_excluded_without_truncation(monkeypatch):
    exact_limit = "x" * platform.MAX_COMMENT_CHARS
    response = Response({"items": [item("   "), item(exact_limit + "y"), item(exact_limit)]})
    mock_get(monkeypatch, response)
    result = platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}", api_key=SECRET_FIXTURE)
    assert result["comments"] == [exact_limit]
    assert result["retrieved_count"] == 1 and result["received_count"] == 3
    assert result["skipped_blank"] == result["skipped_oversized"] == 1
    assert result["has_more"] is False


def test_empty_video_comments_are_successful_empty_subset(monkeypatch):
    mock_get(monkeypatch, Response({"items": []}))
    result = platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}", api_key=SECRET_FIXTURE)
    assert result["comments"] == []
    assert result["received_count"] == result["retrieved_count"] == 0


@pytest.mark.parametrize("reason,status,expected", [
    ("commentsDisabled", 403, "disabled"), ("quotaExceeded", 403, "quota"),
    ("forbidden", 403, "did not allow"), ("videoNotFound", 404, "not found"),
    ("keyInvalid", 400, "local API key"), ("anything-private", 503, "could not be retrieved"),
])
def test_upstream_errors_use_fixed_messages_and_never_echo_key(monkeypatch, reason, status, expected):
    response = Response({"error": {"message": SECRET_FIXTURE, "errors": [{"reason": reason, "message": SECRET_FIXTURE}]}}, status=status)
    mock_get(monkeypatch, response)
    with pytest.raises(platform.PlatformFetchError, match=expected) as failure:
        platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}", api_key=SECRET_FIXTURE)
    assert SECRET_FIXTURE not in str(failure.value)
    assert "anything-private" not in str(failure.value)
    assert response.closed


@pytest.mark.parametrize("failure", [requests.Timeout(SECRET_FIXTURE), requests.ConnectionError(SECRET_FIXTURE)])
def test_request_exceptions_are_sanitized_without_retry(monkeypatch, failure):
    calls = []

    def failing_get(*args, **kwargs):
        calls.append(args)
        raise failure

    monkeypatch.setattr(platform.requests, "get", failing_get)
    with pytest.raises(platform.PlatformFetchError) as caught:
        platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}", api_key=SECRET_FIXTURE)
    assert SECRET_FIXTURE not in str(caught.value)
    assert caught.value.__suppress_context__ is True
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [None, [], {}, {"items": "bad"}, {"items": [None]},
                                     {"items": [{"snippet": {}}]}, {"items": [item(123)]},
                                     {"items": [], "nextPageToken": 123},
                                     {"items": [item("a"), item("b")]}])
def test_malformed_response_is_not_partially_analyzed(monkeypatch, payload):
    response = Response(payload)
    mock_get(monkeypatch, response)
    with pytest.raises(platform.PlatformFetchError, match="unexpected response"):
        platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}", limit=1, api_key=SECRET_FIXTURE)
    assert response.closed


def test_invalid_json_is_safe_and_response_closed(monkeypatch):
    response = Response(body=("invalid " + SECRET_FIXTURE).encode())
    mock_get(monkeypatch, response)
    with pytest.raises(platform.PlatformFetchError) as failure:
        platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}", api_key=SECRET_FIXTURE)
    assert SECRET_FIXTURE not in str(failure.value)
    assert response.closed


@pytest.mark.parametrize("advertised", [True, False])
def test_response_bytes_are_bounded_even_without_content_length(monkeypatch, advertised):
    response = Response(body=b"x" * (platform.MAX_RESPONSE_BYTES + 1),
                        headers={"Content-Length": str(platform.MAX_RESPONSE_BYTES + 1)} if advertised else {})
    mock_get(monkeypatch, response)
    with pytest.raises(platform.PlatformFetchError, match="oversized response"):
        platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}", api_key=SECRET_FIXTURE)
    assert response.closed
    assert response.reads == (0 if advertised else 1)


def test_redirect_is_not_followed(monkeypatch):
    response = Response({"message": SECRET_FIXTURE}, status=302, headers={"Location": "http://127.0.0.1/private"})
    calls = mock_get(monkeypatch, response)
    with pytest.raises(platform.PlatformFetchError):
        platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}", api_key=SECRET_FIXTURE)
    assert len(calls) == 1 and calls[0][1]["allow_redirects"] is False
    assert response.closed


def test_arbitrary_error_reason_is_never_echoed():
    assert SECRET_FIXTURE not in str(platform.PlatformFetchError(SECRET_FIXTURE))
    assert SECRET_FIXTURE not in str(platform.PlatformFetchError([SECRET_FIXTURE]))


def test_malformed_error_reason_cannot_escape_safe_error_handling(monkeypatch):
    response = Response({"error": {"errors": [{"reason": [SECRET_FIXTURE]}]}}, status=403)
    mock_get(monkeypatch, response)
    with pytest.raises(platform.PlatformFetchError, match="did not allow") as failure:
        platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}", api_key=SECRET_FIXTURE)
    assert SECRET_FIXTURE not in str(failure.value)


def test_leading_zero_content_length_remains_bounded_and_parseable(monkeypatch):
    response = Response({"items": []}, headers={"Content-Length": "0" * 5000})
    mock_get(monkeypatch, response)
    result = platform.fetch_youtube_comments(f"https://youtu.be/{VIDEO_ID}", api_key=SECRET_FIXTURE)
    assert result["comments"] == [] and response.closed
