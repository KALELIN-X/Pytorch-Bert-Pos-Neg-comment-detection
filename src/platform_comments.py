"""Optional, bounded retrieval of public top-level YouTube comments.

Uses one official commentThreads.list request. It never fetches the user URL,
follows redirects, paginates, retrieves replies, or stores comments/API keys.
Reference: https://developers.google.com/youtube/v3/docs/commentThreads/list
"""

import json
import os
import re
from urllib.parse import parse_qs, urlsplit

import requests

from .config import MAX_COMMENT_CHARS

YOUTUBE_ENDPOINT = "https://www.googleapis.com/youtube/v3/commentThreads"
MAX_PLATFORM_COMMENTS = 100
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_ERROR_MESSAGES = {
    "unavailable": "YouTube comments could not be retrieved. Try again or paste/upload comments manually.",
    "timeout": "YouTube request timed out. Try again or paste/upload comments manually.",
    "quotaExceeded": "YouTube API quota is exhausted. Try later or paste/upload comments manually.",
    "commentsDisabled": "Comments are disabled for this YouTube video. Try another video or paste/upload comments manually.",
    "forbidden": "YouTube did not allow access to these comments. Check API access or paste/upload comments manually.",
    "videoNotFound": "YouTube video is unavailable or was not found. Check the link or paste/upload comments manually.",
    "bad_request": "YouTube rejected the request. Check the local API key and enable YouTube Data API v3, or paste/upload comments manually.",
    "malformed": "YouTube returned an unexpected response. Try again or paste/upload comments manually.",
    "oversized": "YouTube returned an oversized response. Request fewer comments or paste/upload comments manually.",
}


class InvalidPostURLError(ValueError):
    def __init__(self):
        super().__init__("Enter an HTTPS YouTube watch, shorts, embed, or youtu.be video URL with a valid video ID.")


class MissingPlatformAccessError(RuntimeError):
    def __init__(self):
        super().__init__(
            "YouTube retrieval is not configured. Set YOUTUBE_API_KEY in your ignored local .env "
            "and enable YouTube Data API v3, or paste/upload comments manually."
        )


class PlatformFetchError(RuntimeError):
    """Only fixed public messages are exposed; arbitrary upstream text is ignored."""

    def __init__(self, reason="unavailable"):
        self.reason = reason if isinstance(reason, str) and reason in _ERROR_MESSAGES else "unavailable"
        super().__init__(_ERROR_MESSAGES[self.reason])


def parse_youtube_video_id(post_url):
    """Extract a safe ID from exact allowlisted hosts, without making a request."""
    if type(post_url) is not str or len(post_url) > 2048:
        raise InvalidPostURLError()
    if any(ord(character) < 32 or ord(character) == 127 for character in post_url):
        raise InvalidPostURLError()
    url = post_url.strip()
    if not url or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in url):
        raise InvalidPostURLError()
    try:
        parsed = urlsplit(url)
        # Checking the entire authority rejects ports, user-info and lookalike hosts.
        if parsed.scheme != "https" or parsed.netloc.lower() not in _ALLOWED_HOSTS:
            raise InvalidPostURLError()
        if parsed.netloc.lower() == "youtu.be":
            match = re.fullmatch(r"/([A-Za-z0-9_-]{11})/?", parsed.path)
            video_id = match.group(1) if match else ""
        elif parsed.path == "/watch":
            values = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=20).get("v", [])
            video_id = values[0] if len(values) == 1 else ""
        else:
            match = re.fullmatch(r"/(?:shorts|embed)/([A-Za-z0-9_-]{11})/?", parsed.path)
            video_id = match.group(1) if match else ""
        if not _VIDEO_ID.fullmatch(video_id):
            raise InvalidPostURLError()
    except (ValueError, UnicodeError):
        raise InvalidPostURLError() from None
    return video_id


def _read_bounded_json(response):
    length = response.headers.get("Content-Length", "")
    if isinstance(length, str) and length.isdigit():
        digits = length.lstrip("0") or "0"
        if len(digits) > 10 or int(digits) > MAX_RESPONSE_BYTES:
            raise PlatformFetchError("oversized")
    body = bytearray()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        if not isinstance(chunk, bytes):
            raise PlatformFetchError("malformed")
        if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
            raise PlatformFetchError("oversized")
        body.extend(chunk)
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        raise PlatformFetchError("malformed") from None
    if not isinstance(payload, dict):
        raise PlatformFetchError("malformed")
    return payload


def _raise_api_error(status, payload):
    reason = None
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("errors"), list):
        for detail in error["errors"]:
            if isinstance(detail, dict) and isinstance(detail.get("reason"), str) and detail["reason"] in {
                "commentsDisabled", "quotaExceeded", "forbidden", "videoNotFound",
            }:
                reason = detail["reason"]
                break
    if reason is None:
        reason = {400: "bad_request", 401: "forbidden", 403: "forbidden", 404: "videoNotFound", 429: "quotaExceeded"}.get(status, "unavailable")
    raise PlatformFetchError(reason)


def fetch_youtube_comments(post_url, *, limit=MAX_PLATFORM_COMMENTS, api_key=None):
    """Retrieve at most one page of newest top-level comments.

    `retrieved_count` counts retained comments. `received_count` includes blank
    and oversized exclusions. Text is never truncated. `has_more` indicates
    a page token, which is deliberately not followed or returned.
    """
    video_id = parse_youtube_video_id(post_url)
    if type(limit) is not int or not 1 <= limit <= MAX_PLATFORM_COMMENTS:
        raise ValueError("YouTube comment limit must be an integer between 1 and 100.")
    configured_key = api_key if api_key is not None else os.getenv("YOUTUBE_API_KEY", "")
    if not isinstance(configured_key, str) or not configured_key.strip():
        raise MissingPlatformAccessError()
    response = None
    try:
        # No user-controlled URL reaches the HTTP client. No automatic redirects/retries.
        response = requests.get(
            YOUTUBE_ENDPOINT,
            params={"part": "snippet", "videoId": video_id, "maxResults": limit,
                    "order": "time", "textFormat": "plainText", "key": configured_key.strip()},
            headers={"Accept": "application/json"}, timeout=(5, 20),
            allow_redirects=False, stream=True,
        )
        payload = _read_bounded_json(response)
        if response.status_code != 200:
            _raise_api_error(response.status_code, payload)
    except requests.Timeout:
        raise PlatformFetchError("timeout") from None
    except requests.RequestException:
        # Request exceptions can contain the API-key query string.
        raise PlatformFetchError("unavailable") from None
    finally:
        if response is not None:
            try:
                response.close()
            except requests.RequestException:
                pass
    items = payload.get("items")
    next_page = payload.get("nextPageToken", "")
    if not isinstance(items, list) or len(items) > limit or not isinstance(next_page, str):
        raise PlatformFetchError("malformed")
    comments, skipped_blank, skipped_oversized = [], 0, 0
    for item in items:
        try:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        except (KeyError, TypeError):
            raise PlatformFetchError("malformed") from None
        if not isinstance(comment, str):
            raise PlatformFetchError("malformed")
        if not comment.strip():
            skipped_blank += 1
        elif len(comment) > MAX_COMMENT_CHARS:
            skipped_oversized += 1
        else:
            comments.append(comment)
    return {
        "comments": comments, "source": "youtube", "scope": "retrieved_top_level_comments",
        "video_id": video_id, "retrieved_count": len(comments), "received_count": len(items),
        "skipped_blank": skipped_blank, "skipped_oversized": skipped_oversized, "has_more": bool(next_page),
    }
