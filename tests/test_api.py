"""HTTP contract tests use an injected evaluator; no downloads or real data."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from src.api import BodySizeLimitMiddleware, create_app as build_app
from src.config import MAX_REQUEST_BYTES, POSITIVE_LABELS, TOXIC_LABELS
from src.inference import InputTooLongError, ModelUnavailableError, PredictionError
from src.platform_comments import InvalidPostURLError, MissingPlatformAccessError, PlatformFetchError


def example_result(positive_score=0.8):
    positivity = {label: positive_score for label in POSITIVE_LABELS}
    toxicity = {label: 0.1 for label in TOXIC_LABELS}
    return {
        "toxicity": toxicity, "positivity": positivity,
        "toxicity_bin": {label: 0 for label in TOXIC_LABELS},
        "positivity_bin": {label: 1 for label in POSITIVE_LABELS},
        "net_goodness": positive_score - 1.2 * 0.1,
        "label": "positive_signals", "explanation": "Positive emotion signals meet the selected threshold.",
    }


class FakeEvaluator:
    """Deterministic test adapter, deliberately not a sentiment classifier."""

    def __init__(self, load_error=None, prediction_error=None, outputs=None):
        self.ready = False
        self.load_error = load_error
        self.prediction_error = prediction_error
        self.outputs = outputs
        self.load_calls = 0
        self.calls = []

    def load(self):
        self.load_calls += 1
        if self.load_error:
            raise self.load_error
        self.ready = True

    def predict(self, texts, **options):
        self.calls.append((texts, options))
        if self.prediction_error:
            raise self.prediction_error
        return self.outputs if self.outputs is not None else [example_result(max(0.01, 0.8 - index * 0.1)) for index, _ in enumerate(texts)]


def sentiment_result(label="positive"):
    return {
        "label": label, "scores": {name: 0.8 if name == label else 0.1 for name in ("positive", "neutral", "negative")},
        "explanation": f"The sentence sentiment model predicts {label}. Scores are uncalibrated.",
    }


class FakeSentimentEvaluator(FakeEvaluator):
    setup_command = "python -m src.download_sentiment"
    max_tokens = 192

    def predict(self, texts):
        self.calls.append(texts)
        if self.prediction_error:
            raise self.prediction_error
        return self.outputs if self.outputs is not None else [sentiment_result() for _ in texts]


def create_app(evaluator=None, sentiment_evaluator=None, **options):
    # Always inject both engines so these tests never load cached real artifacts.
    return build_app(evaluator if evaluator is not None else FakeEvaluator(),
                     sentiment_evaluator if sentiment_evaluator is not None else FakeSentimentEvaluator(), **options)


@pytest.fixture
def api_client():
    evaluator = FakeEvaluator()
    with TestClient(create_app(evaluator)) as client:
        yield client, evaluator


def test_single_preserves_original_fields_and_explains_scores(api_client):
    client, evaluator = api_client
    response = client.post("/predict", json={"text": "  Thanks for helping.  "})
    assert response.status_code == 200
    result = response.json()
    assert {"toxicity", "toxicity_bin", "positivity", "positivity_bin", "net_goodness"} <= result.keys()
    assert result["label"] == "positive_signals"
    assert result["explanation"]
    assert result["metadata"]["confidence_calibrated"] is False
    assert result["metadata"]["score_type"] == "uncalibrated_model_scores"
    assert result["metadata"]["max_tokens"] == 192
    assert "heuristic" in result["metadata"]["description"]
    assert "confidence" not in result
    assert "text" not in result
    assert evaluator.calls == [(["Thanks for helping."], {"threshold": 0.5, "w_pos": 1.0, "w_tox": 1.2})]


def test_batch_returns_one_ordered_result_per_comment(api_client):
    client, evaluator = api_client
    response = client.post("/predict", json={"texts": ["  First  ", "Second", "Third"]})
    assert response.status_code == 200
    result = response.json()
    assert result["count"] == 3
    assert [item["index"] for item in result["results"]] == [0, 1, 2]
    assert [item["positivity"]["joy"] for item in result["results"]] == pytest.approx([0.8, 0.7, 0.6])
    assert evaluator.calls[0][0] == ["First", "Second", "Third"]


def test_maximum_batch_size_and_character_bound_are_accepted(api_client):
    client, evaluator = api_client
    response = client.post("/predict", json={"texts": ["x" * 2000] * 20})
    assert response.status_code == 200
    assert response.json()["count"] == 20
    assert len(evaluator.calls[0][0]) == 20


def test_health_and_multiple_calls_load_models_only_once(api_client):
    client, evaluator = api_client
    health = client.get("/health").json()
    assert health["ready"] is True
    assert health["limits"] == {"max_comment_chars": 2000, "max_comments": 20, "max_tokens": 192, "max_request_bytes": 65536}
    client.post("/predict", json={"text": "One"})
    client.post("/predict", json={"text": "Two"})
    assert evaluator.load_calls == 1


def test_landing_and_interactive_api_document_the_actual_task(api_client):
    client, _ = api_client
    landing = client.get("/")
    assert landing.status_code == 200
    assert "six positive emotions" in landing.text
    assert 'href="/docs"' in landing.text
    assert 'href="/health"' in landing.text
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "not binary sentiment" in schema["info"]["description"]
    assert schema["components"]["schemas"]["PredictionRequest"]["additionalProperties"] is False


def test_sentiment_single_is_distinct_from_original_emotion_workflow(api_client):
    client, _ = api_client
    response = client.post("/sentiment", json={"text": "  A helpful explanation.  "})
    assert response.status_code == 200
    result = response.json()
    assert result["label"] == "positive"
    assert set(result["scores"]) == {"positive", "neutral", "negative"}
    assert "toxicity" not in result and "net_goodness" not in result
    assert result["metadata"]["confidence_calibrated"] is False
    assert "confidence" not in result


def test_sentiment_batch_percentages_count_predicted_labels_not_average_scores():
    evaluator = FakeSentimentEvaluator(outputs=[sentiment_result(label) for label in ("positive", "negative", "neutral", "positive")])
    with TestClient(create_app(sentiment_evaluator=evaluator)) as client:
        response = client.post("/sentiment", json={"texts": ["One", "Two", "Three", "Four"]})
        assert response.status_code == 200
        result = response.json()
        assert result["count"] == 4
        assert [item["index"] for item in result["results"]] == [0, 1, 2, 3]
        assert result["aggregate"] == {"total": 4, "counts": {"positive": 2, "neutral": 1, "negative": 1},
                                       "percentages": {"positive": 50.0, "neutral": 25.0, "negative": 25.0}}
        assert "analyzed comments" in result["metadata"]["description"]
        assert evaluator.calls == [["One", "Two", "Three", "Four"]]


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": 5}, {"texts": []},
                                     {"texts": ["Okay", " "]}, {"texts": ["x"] * 21}, {"text": "x" * 2001},
                                     {"text": "One", "texts": ["Two"]}, {"text": "Okay", "threshold": 0.5}])
def test_sentiment_request_bounds_and_no_legacy_scoring_options(payload):
    evaluator = FakeSentimentEvaluator()
    with TestClient(create_app(sentiment_evaluator=evaluator)) as client:
        response = client.post("/sentiment", json=payload)
        assert response.status_code == 422
        assert not evaluator.calls


def test_sentiment_model_unavailable_does_not_disable_original_predict():
    sentiment = FakeSentimentEvaluator(load_error=ModelUnavailableError("private-path"))
    with TestClient(create_app(sentiment_evaluator=sentiment)) as client:
        assert client.get("/health").json()["sentiment_ready"] is False
        response = client.post("/sentiment", json={"text": "Comment"})
        assert response.status_code == 503
        assert response.json()["detail"]["action"] == "python -m src.download_sentiment"
        assert "private-path" not in response.text
        assert client.post("/predict", json={"text": "Comment"}).status_code == 200


def test_original_model_unavailable_does_not_disable_sentiment():
    with TestClient(create_app(evaluator=FakeEvaluator(load_error=ModelUnavailableError("private")))) as client:
        health = client.get("/health").json()
        assert health["ready"] is False and health["sentiment_ready"] is True
        assert client.post("/sentiment", json={"text": "Comment"}).status_code == 200


@pytest.mark.parametrize("error,expected_status,code", [(PredictionError("private-comment"), 503, "prediction_failed"),
                                                      (ModelUnavailableError("private-path"), 503, "models_unavailable"),
                                                      (InputTooLongError(0, 192), 422, "comment_too_many_tokens")])
def test_sentiment_failure_statuses_are_clear_and_safe(error, expected_status, code):
    with TestClient(create_app(sentiment_evaluator=FakeSentimentEvaluator(prediction_error=error))) as client:
        response = client.post("/sentiment", json={"text": "Comment"})
        assert response.status_code == expected_status
        assert response.json()["detail"]["code"] == code
        assert "private" not in response.text


@pytest.mark.parametrize("kind", ["missing_score", "nonfinite", "not_normalized", "wrong_label", "wrong_count"])
def test_invalid_sentiment_outputs_are_rejected(kind):
    result = sentiment_result()
    outputs = [result]
    if kind == "missing_score":
        del result["scores"]["neutral"]
    elif kind == "nonfinite":
        result["scores"]["positive"] = float("nan")
    elif kind == "not_normalized":
        result["scores"] = {"positive": 0.9, "neutral": 0.9, "negative": 0.9}
    elif kind == "wrong_label":
        result["label"] = "negative"
    elif kind == "wrong_count":
        outputs = []
    with TestClient(create_app(sentiment_evaluator=FakeSentimentEvaluator(outputs=outputs))) as client:
        assert client.post("/sentiment", json={"text": "Comment"}).status_code == 503


def youtube_result(comments=None):
    comments = ["Helpful explanation.", "I disagree."] if comments is None else comments
    return {"comments": comments, "source": "youtube", "scope": "retrieved_top_level_comments",
            "retrieved_count": len(comments), "received_count": len(comments), "skipped_blank": 0,
            "skipped_oversized": 0, "has_more": False, "video_id": "AbCdEfGhIjK"}


def test_youtube_route_is_bounded_and_keeps_credentials_private(monkeypatch):
    from unittest.mock import Mock
    import src.api as api

    monkeypatch.setattr(api, "YOUTUBE_API_KEY", "test-only-placeholder")
    fetch = Mock(return_value=youtube_result())
    with TestClient(create_app(youtube_fetcher=fetch)) as client:
        assert client.get("/health").json()["youtube_configured"] is True
        response = client.post("/post-comments/youtube", json={"url": "https://www.youtube.com/watch?v=AbCdEfGhIjK", "max_comments": 20})
        assert response.status_code == 200
        assert response.json()["retrieved_count"] == 2
        assert response.json()["scope"] == "retrieved_top_level_comments"
        assert "test-only-placeholder" not in response.text
        assert fetch.call_args.kwargs == {"limit": 20, "api_key": "test-only-placeholder"}


@pytest.mark.parametrize("payload", [{}, {"url": ""}, {"url": "  "}, {"url": 5}, {"url": "x" * 2049},
                                     {"url": "url", "max_comments": 0}, {"url": "url", "max_comments": 101},
                                     {"url": "url", "max_comments": True}, {"url": "url", "max_comments": "20"},
                                     {"url": "url", "private-key": "private-value"}])
def test_youtube_invalid_request_never_calls_upstream(payload):
    from unittest.mock import Mock

    fetch = Mock(return_value=youtube_result())
    with TestClient(create_app(youtube_fetcher=fetch)) as client:
        response = client.post("/post-comments/youtube", json=payload)
        assert response.status_code == 422
        assert not fetch.called
        assert "private" not in response.text


@pytest.mark.parametrize("error,status,code", [(InvalidPostURLError(), 422, "invalid_post_url"),
                                              (MissingPlatformAccessError(), 503, "youtube_not_configured"),
                                              (PlatformFetchError("private-upstream"), 502, "youtube_fetch_failed"),
                                              (RuntimeError("private-stack"), 502, "youtube_fetch_failed")])
def test_youtube_failure_details_do_not_expose_upstream_data(error, status, code):
    from unittest.mock import Mock

    with TestClient(create_app(youtube_fetcher=Mock(side_effect=error))) as client:
        response = client.post("/post-comments/youtube", json={"url": "https://youtu.be/AbCdEfGhIjK"})
        assert response.status_code == status
        assert response.json()["detail"]["code"] == code
        assert "private" not in response.text


def test_youtube_empty_thread_has_valid_zero_counts():
    from unittest.mock import Mock

    with TestClient(create_app(youtube_fetcher=Mock(return_value=youtube_result([])))) as client:
        response = client.post("/post-comments/youtube", json={"url": "https://youtu.be/AbCdEfGhIjK"})
        assert response.status_code == 200
        assert response.json()["comments"] == []
        assert response.json()["received_count"] == 0


def test_youtube_invalid_or_over_limit_upstream_output_is_sanitized():
    from unittest.mock import Mock

    with TestClient(create_app(youtube_fetcher=Mock(return_value=youtube_result()))) as client:
        response = client.post("/post-comments/youtube", json={"url": "https://youtu.be/AbCdEfGhIjK", "max_comments": 1})
        assert response.status_code == 502
        assert "Helpful explanation" not in response.text


@pytest.mark.parametrize("reason,status,phrase", [("quotaExceeded", 503, "quota"), ("timeout", 503, "timed out"),
                                                 ("commentsDisabled", 502, "disabled")])
def test_youtube_whitelisted_upstream_reasons_remain_actionable(reason, status, phrase):
    from unittest.mock import Mock

    with TestClient(create_app(youtube_fetcher=Mock(side_effect=PlatformFetchError(reason)))) as client:
        response = client.post("/post-comments/youtube", json={"url": "https://youtu.be/AbCdEfGhIjK"})
        assert response.status_code == status
        assert phrase in response.json()["detail"]["message"]
        assert response.json()["detail"]["reason"] == reason


@pytest.mark.parametrize("payload", [
    {}, {"text": ""}, {"text": " \t\n "}, {"text": None},
    {"text": 7}, {"text": True}, {"text": ["One"]}, {"text": {"comment": "One"}},
    {"text": "x" * 2001}, {"text": " " * 2001},
    {"texts": None}, {"texts": []}, {"texts": "One"}, {"texts": ["One", " "]},
    {"texts": ["One", None]}, {"texts": ["One", 3]}, {"texts": ["One"] * 21},
    {"texts": ["One", "x" * 2001]}, {"text": "One", "texts": ["Two"]},
    {"text": "One", "extra": "private"}, ["One"], "One", 9,
])
def test_invalid_inputs_are_rejected_before_prediction(api_client, payload):
    client, evaluator = api_client
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert not evaluator.calls
    assert all("input" not in detail and "ctx" not in detail for detail in response.json()["detail"])


def test_batch_validation_identifies_the_invalid_item(api_client):
    client, _ = api_client
    response = client.post("/predict", json={"texts": ["Valid", "   "]})
    assert response.json()["detail"][0]["path"] == "texts.1"


def test_malformed_unicode_comment_is_rejected(api_client):
    client, evaluator = api_client
    # Escape the surrogate in JSON: HTTPx rejects it before sending otherwise.
    response = client.post("/predict", content='{"text":"\\ud800"}', headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "unicode" in response.text.lower()
    assert not evaluator.calls


@pytest.mark.parametrize("field,value", [
    ("threshold", -0.01), ("threshold", 1.01), ("threshold", True), ("threshold", "0.5"),
    ("w_pos", -0.01), ("w_pos", 3.01), ("w_pos", True), ("w_pos", "1"),
    ("w_tox", -0.01), ("w_tox", 3.01), ("w_tox", False), ("w_tox", "1.2"),
])
def test_invalid_numeric_options_are_rejected(api_client, field, value):
    client, evaluator = api_client
    response = client.post("/predict", json={"text": "Hello", field: value})
    assert response.status_code == 422
    assert not evaluator.calls


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("field", ["threshold", "w_pos", "w_tox"])
def test_non_finite_options_are_rejected(api_client, number, field):
    client, evaluator = api_client
    response = client.post("/predict", content='{"text":"Hello","' + field + '":' + number + '}', headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert not evaluator.calls


@pytest.mark.parametrize("threshold", [0, 1])
def test_endpoint_thresholds_and_weight_bounds_remain_supported(api_client, threshold):
    client, evaluator = api_client
    response = client.post("/predict", json={"text": "Hello", "threshold": threshold, "w_pos": 0, "w_tox": 3})
    assert response.status_code == 200
    assert evaluator.calls[0][1] == {"threshold": threshold, "w_pos": 0, "w_tox": 3}


@pytest.mark.parametrize("body", ['{"text":"private', "", "null", "{", "[1,2]"])
def test_malformed_json_or_non_object_does_not_echo_input(api_client, body):
    client, evaluator = api_client
    response = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "private" not in response.text
    assert not evaluator.calls


def test_unknown_key_and_invalid_value_remain_private(api_client):
    client, _ = api_client
    response = client.post("/predict", json={"text": "public", "private-secret-as-key": "private-secret-value"})
    assert response.status_code == 422
    assert "private-secret" not in response.text
    response = client.post("/predict", json={"text": {"private-secret": "private-secret-value"}})
    assert "private-secret" not in response.text


def test_oversized_body_rejected_without_running_prediction(api_client):
    client, evaluator = api_client
    response = client.post("/predict", content=b" " * (MAX_REQUEST_BYTES + 1), headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert str(MAX_REQUEST_BYTES) in response.text
    assert not evaluator.calls


@pytest.mark.parametrize("error", [ModelUnavailableError("private-path-secret"), RuntimeError("private-path-secret")])
def test_missing_or_broken_models_do_not_prevent_startup(error):
    evaluator = FakeEvaluator(load_error=error)
    with TestClient(create_app(evaluator)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["ready"] is False
        assert health.json()["action"] == "python -m src.download_models"
        response = client.post("/predict", json={"text": "Demo comment"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "models_unavailable"
        assert "private-path-secret" not in response.text
        assert not evaluator.calls


def test_shared_backend_recovery_command_is_not_a_public_model_download():
    evaluator = FakeEvaluator(load_error=ModelUnavailableError("private"))
    evaluator.setup_command = "python -m src.train_multitask --help"
    with TestClient(create_app(evaluator)) as client:
        response = client.post("/predict", json={"text": "Comment"})
        assert response.json()["detail"]["action"] == "python -m src.train_multitask --help"
        assert "download_models" not in response.text


def test_shared_backend_health_reports_its_actual_token_budget():
    evaluator = FakeEvaluator()
    evaluator.backend = "shared_bert"
    evaluator.max_tokens = 128
    with TestClient(create_app(evaluator)) as client:
        assert client.get("/health").json()["limits"]["max_tokens"] == 128
        response = client.post("/predict", json={"text": "Comment"})
        assert response.json()["metadata"]["max_tokens"] == 128
        assert response.json()["metadata"]["backend"] == "shared_bert"


@pytest.mark.parametrize("error", [PredictionError("private-text-secret"), RuntimeError("private-text-secret")])
def test_prediction_failures_have_clear_safe_errors(error):
    with TestClient(create_app(FakeEvaluator(prediction_error=error))) as client:
        response = client.post("/predict", json={"text": "Comment"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "prediction_failed"
        assert "private-text-secret" not in response.text


def test_lost_model_marks_health_unavailable():
    with TestClient(create_app(FakeEvaluator(prediction_error=ModelUnavailableError("private")))) as client:
        response = client.post("/predict", json={"text": "Comment"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "models_unavailable"
        assert client.get("/health").json()["ready"] is False


@pytest.mark.parametrize("payload,path,index", [({"text": "Comment"}, "text", 0), ({"texts": ["First", "Second"]}, "texts.1", 1)])
def test_token_limit_failure_is_actionable_and_does_not_silently_truncate(payload, path, index):
    evaluator = FakeEvaluator(prediction_error=InputTooLongError(index=index, max_tokens=192))
    with TestClient(create_app(evaluator)) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["code"] == "comment_too_many_tokens"
        assert detail["path"] == path
        assert "192" in detail["message"]
        assert "not truncated" in detail["message"]


@pytest.mark.parametrize("kind", ["missing_label", "unknown_label", "nonfinite", "out_of_range", "invalid_bin", "unknown_summary", "wrong_count"])
def test_invalid_model_outputs_are_not_returned(kind):
    result = deepcopy(example_result())
    outputs = [result]
    if kind == "missing_label":
        del result["toxicity"]["threat"]
    elif kind == "unknown_label":
        result["toxicity"]["other"] = 0.3
    elif kind == "nonfinite":
        result["net_goodness"] = float("nan")
    elif kind == "out_of_range":
        result["positivity"]["joy"] = 1.2
    elif kind == "invalid_bin":
        result["toxicity_bin"]["threat"] = True
    elif kind == "unknown_summary":
        result["label"] = "certainly_positive"
    elif kind == "wrong_count":
        outputs = []
    with TestClient(create_app(FakeEvaluator(outputs=outputs))) as client:
        response = client.post("/predict", json={"text": "Comment"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "prediction_failed"


def test_body_limit_handles_chunked_stream_without_content_length():
    """Exercise the middleware directly: TestClient normally inserts a length."""
    import asyncio

    called = []
    sent = []
    chunks = iter([
        {"type": "http.request", "body": b"x" * (MAX_REQUEST_BYTES - 1), "more_body": True},
        {"type": "http.request", "body": b"xx", "more_body": False},
    ])

    async def inner(scope, receive, send):
        called.append(True)

    async def receive():
        return next(chunks)

    async def send(message):
        sent.append(message)

    asyncio.run(BodySizeLimitMiddleware(inner)({"type": "http", "method": "POST", "headers": []}, receive, send))
    assert not called
    assert sent[0]["status"] == 413
