"""Exercise the local dashboard with mocked HTTP responses and no model access."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.config import POSITIVE_LABELS, TOXIC_LABELS

DASHBOARD = Path(__file__).resolve().parents[1] / "MainFolder" / "app" / "Dashboard.py"


class Response:
    def __init__(self, data, status_code=200):
        self.data = data
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self):
        return deepcopy(self.data)

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError("Test response")


def prediction():
    return {
        "toxicity": {label: 0.1 for label in TOXIC_LABELS},
        "positivity": {label: 0.8 for label in POSITIVE_LABELS},
        "toxicity_bin": {label: 0 for label in TOXIC_LABELS},
        "positivity_bin": {label: 1 for label in POSITIVE_LABELS},
        "net_goodness": 0.68, "label": "positive_signals",
        "explanation": "Positive emotion signals meet the selected threshold.",
    }


def metadata():
    return {
        "threshold": 0.5, "w_pos": 1.0, "w_tox": 1.2,
        "score_type": "uncalibrated_model_scores", "confidence_calibrated": False,
        "backend": "public", "max_tokens": 192, "description": "Uncalibrated model scores.",
    }


def find_button(application, label):
    return next(button for button in application.button if button.label == label)


def find_radio(application, label):
    return next(radio for radio in application.radio if radio.label == label)


@pytest.fixture
def dashboard(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:8000/predict")
    get = Mock(return_value=Response({"ready": True, "sentiment_ready": True, "youtube_configured": False,
                                     "backend": "public", "limits": {"max_tokens": 192}}))
    post = Mock(return_value=Response({**prediction(), "metadata": metadata()}))
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    application = AppTest.from_file(str(DASHBOARD), default_timeout=15).run()
    find_radio(application, "What to evaluate").set_value("Emotions & toxicity").run()
    find_button(application, "Appreciation").click().run()
    assert not application.exception
    return application, get, post


def test_initial_page_and_examples_require_no_prediction(dashboard):
    application, get, post = dashboard
    assert application.title[0].value == "Comment evaluator"
    assert "Models ready" in [item.value for item in application.success]
    assert application.text_area[0].value.startswith("Thank you")
    assert get.call_args.args == ("http://127.0.0.1:8000/health",)
    assert get.call_args.kwargs["timeout"] == 2
    assert not post.called


@pytest.mark.parametrize("example,prefix", [("Appreciation", "Thank you"), ("Hostility", "You are"), ("Factual", "The meeting"), ("Ambiguous", "Great, another")])
def test_example_buttons_fill_text_without_claiming_a_prediction(dashboard, example, prefix):
    application, _, post = dashboard
    find_button(application, example).click().run()
    assert application.text_area[0].value.startswith(prefix)
    assert not post.called
    assert not application.exception


def test_single_comment_sends_settings_and_renders_api_result(dashboard):
    application, _, post = dashboard
    application.text_area[0].input("Thanks for the explanation.")
    find_button(application, "Evaluate comment").click().run()
    assert not application.exception
    assert post.call_args.kwargs == {
        "json": {"text": "Thanks for the explanation.", "threshold": 0.5, "w_pos": 1.0, "w_tox": 1.2},
        "timeout": (5, 120),
    }
    assert any("Positive emotion signals" in item.value for item in application.markdown)
    assert any("uncalibrated" in item.value and "not confidence" in item.value for item in application.caption)
    assert any("Evaluated text: Thanks for the explanation." == item.value for item in application.caption)
    assert len(application.json) == 1


def test_sidebar_options_are_sent_to_the_api(dashboard):
    application, _, post = dashboard
    application.slider[0].set_value(0.7)
    application.number_input[0].set_value(2.0)
    application.number_input[1].set_value(0.8)
    application.run()
    find_button(application, "Evaluate comment").click().run()
    assert not application.exception
    payload = post.call_args.kwargs["json"]
    assert payload["threshold"] == 0.7
    assert payload["w_pos"] == 2.0
    assert payload["w_tox"] == 0.8


def test_batch_skips_blank_separator_lines_and_keeps_result_order(dashboard):
    application, _, post = dashboard
    first = prediction()
    second = {**prediction(), "label": "no_detected_signals", "explanation": "No selected signal meets the threshold."}
    post.return_value = Response({"results": [{"index": 0, **first}, {"index": 1, **second}], "count": 2, "metadata": metadata()})
    find_radio(application, "Evaluation mode").set_value("Post comments").run()
    application.text_area[0].input("  First comment  \n\n Second comment\n ")
    find_button(application, "Evaluate comments").click().run()
    assert not application.exception
    assert post.call_args.kwargs["json"]["texts"] == ["First comment", "Second comment"]
    assert [item.value for item in application.caption if item.value.startswith("Evaluated text:")] == [
        "Evaluated text: First comment", "Evaluated text: Second comment",
    ]
    assert len(application.json) == 2


@pytest.mark.parametrize("text,message", [(" \n\t ", "nonblank"), ("x" * 2001, "at most 2,000 characters")], ids=["blank", "too-long"])
def test_invalid_single_input_does_not_call_prediction(dashboard, text, message):
    application, _, post = dashboard
    application.text_area[0].input(text)
    find_button(application, "Evaluate comment").click().run()
    assert not application.exception
    assert any(message in item.value for item in application.error)
    assert not post.called


def test_batch_count_limit_is_checked_locally(dashboard):
    application, _, post = dashboard
    find_radio(application, "Evaluation mode").set_value("Post comments").run()
    application.text_area[0].input("\n".join(["Comment"] * 21))
    find_button(application, "Evaluate comments").click().run()
    assert not application.exception
    assert any("at most 20" in item.value for item in application.error)
    assert not post.called


@pytest.mark.parametrize("failure,message", [(requests.ConnectionError("Test offline"), "Cannot reach the API"), (requests.Timeout("Test timeout"), "took too long")])
def test_network_and_timeout_errors_are_readable(dashboard, failure, message):
    application, _, post = dashboard
    post.side_effect = failure
    find_button(application, "Evaluate comment").click().run()
    assert not application.exception
    assert any(message in item.value for item in application.error)


def test_model_setup_failure_is_actionable(dashboard):
    application, get, post = dashboard
    get.return_value = Response({"ready": False, "action": "python -m src.download_models"})
    post.return_value = Response({"detail": {"message": "Local model artifacts are unavailable.", "action": "python -m src.download_models"}}, status_code=503)
    application.run()
    assert "Models need setup" in [item.value for item in application.warning]
    find_button(application, "Evaluate comment").click().run()
    assert not application.exception
    assert any("python -m src.download_models" in item.value for item in application.error)


def test_api_validation_error_is_readable(dashboard):
    application, _, post = dashboard
    post.return_value = Response({"detail": [{"path": "text", "message": "Shorten this comment to at most 192 model tokens."}]}, status_code=422)
    find_button(application, "Evaluate comment").click().run()
    assert not application.exception
    assert any("text:" in item.value and "192" in item.value for item in application.error)


def test_offline_health_check_does_not_break_the_dashboard(dashboard):
    application, get, _ = dashboard
    get.side_effect = requests.ConnectionError("Test offline")
    application.run()
    assert not application.exception
    assert any("API offline" in item.value for item in application.caption)


@pytest.mark.parametrize("data", [{"metadata": metadata()}, {**prediction(), "metadata": {}}, {"results": [], "count": 0, "metadata": metadata()}])
def test_incomplete_success_response_does_not_crash_or_display_results(dashboard, data):
    application, _, post = dashboard
    post.return_value = Response(data)
    find_button(application, "Evaluate comment").click().run()
    assert not application.exception
    assert any("incomplete result" in item.value for item in application.error)
    assert not application.json


def sentiment_prediction(label="positive"):
    return {"label": label, "scores": {name: 0.8 if name == label else 0.1 for name in ("positive", "neutral", "negative")},
            "explanation": f"Sentence sentiment prediction: {label}."}


def sentiment_metadata():
    return {"score_type": "uncalibrated_model_scores", "confidence_calibrated": False,
            "max_tokens": 192, "description": "Scores are uncalibrated; percentages describe analyzed comments."}


def sentiment_batch(labels):
    counts = {label: labels.count(label) for label in ("positive", "neutral", "negative")}
    return {"results": [{"index": index, **sentiment_prediction(label)} for index, label in enumerate(labels)],
            "count": len(labels), "aggregate": {"total": len(labels), "counts": counts,
                                                  "percentages": {label: count / len(labels) * 100 for label, count in counts.items()}},
            "metadata": sentiment_metadata()}


@pytest.fixture
def sentiment_dashboard(dashboard):
    application, get, post = dashboard
    find_radio(application, "What to evaluate").set_value("Sentiment").run()
    post.return_value = Response({**sentiment_prediction(), "metadata": sentiment_metadata()})
    return application, get, post


def test_sentiment_is_default_and_uses_separate_route(sentiment_dashboard):
    application, _, post = sentiment_dashboard
    assert find_radio(application, "What to evaluate").options[0] == "Sentiment"
    application.text_area[0].input("I enjoyed this explanation.")
    find_button(application, "Evaluate comment").click().run()
    assert not application.exception
    assert post.call_args.args == ("http://127.0.0.1:8000/sentiment",)
    assert post.call_args.kwargs["json"] == {"text": "I enjoyed this explanation."}
    assert any("Positive sentiment" in item.value for item in application.markdown)
    assert any("separate from" in item.value for item in application.caption)


def test_post_percentages_aggregate_all_21_comments_across_two_bounded_requests(sentiment_dashboard):
    application, _, post = sentiment_dashboard
    post.side_effect = [Response(sentiment_batch(["positive"] * 20)), Response(sentiment_batch(["negative"]))]
    find_radio(application, "Evaluation mode").set_value("Post comments").run()
    application.text_area[0].input("\n".join(f"Comment {index}" for index in range(21)))
    find_button(application, "Evaluate comments").click().run()
    assert not application.exception
    assert [len(call.kwargs["json"]["texts"]) for call in post.call_args_list] == [20, 1]
    assert [metric.value for metric in application.metric] == ["95.2%", "0.0%", "4.8%"]
    assert any("21 comments analyzed" in item.value for item in application.caption)
    assert any("not confidence" in item.value for item in application.caption)
    assert len(application.dataframe[0].value) == 21


def test_post_partial_prediction_failure_does_not_publish_partial_totals(sentiment_dashboard):
    application, _, post = sentiment_dashboard
    post.side_effect = [Response(sentiment_batch(["positive"] * 20)),
                        Response({"detail": {"message": "Sentiment model is unavailable."}}, status_code=503)]
    find_radio(application, "Evaluation mode").set_value("Post comments").run()
    application.text_area[0].input("\n".join(["Comment"] * 21))
    find_button(application, "Evaluate comments").click().run()
    assert not application.exception
    assert any("unavailable" in item.value for item in application.error)
    assert not application.metric
    assert list(application.dataframe[0].value.columns) == ['Comment number', 'Text']
    assert len(application.dataframe[0].value) == 21


def test_post_token_error_uses_global_comment_number_and_keeps_inputs_for_review(sentiment_dashboard):
    application, _, post = sentiment_dashboard
    post.side_effect = [Response(sentiment_batch(['positive'] * 20)),
                        Response({'detail': {'path': 'texts.3', 'message': 'Shorten this comment to at most 192 tokens.'}}, status_code=422)]
    find_radio(application, 'Evaluation mode').set_value('Post comments').run()
    application.text_area[0].input('\n'.join(f'Comment text {index + 1}' for index in range(24)))
    find_button(application, 'Evaluate comments').click().run()
    assert not application.exception
    assert any('Comment 24:' in item.value for item in application.error)
    assert not application.metric
    assert application.dataframe[0].value.iloc[23].to_dict() == {'Comment number': 24, 'Text': 'Comment text 24'}


def test_manual_post_import_limit_is_100_comments(sentiment_dashboard):
    application, _, post = sentiment_dashboard
    find_radio(application, "Evaluation mode").set_value("Post comments").run()
    application.text_area[0].input("\n".join(["Comment"] * 101))
    find_button(application, "Evaluate comments").click().run()
    assert not application.exception
    assert any("at most 100" in item.value for item in application.error)
    assert not post.called


def test_csv_mode_requires_a_file_before_prediction(sentiment_dashboard):
    application, _, post = sentiment_dashboard
    find_radio(application, "Evaluation mode").set_value("Post comments").run()
    find_radio(application, "Comment source").set_value("CSV upload").run()
    find_button(application, "Evaluate comments").click().run()
    assert not application.exception
    assert any("Choose a comments CSV" in item.value for item in application.error)
    assert not post.called


@pytest.mark.parametrize("status,message", [(503, "YouTube needs a local API key."), (404, "YouTube video is unavailable."),
                                           (502, "YouTube quota is exhausted.")])
def test_youtube_errors_remain_readable_and_do_not_start_prediction(sentiment_dashboard, status, message):
    application, _, post = sentiment_dashboard
    post.return_value = Response({"detail": {"message": message}}, status_code=status)
    find_radio(application, "Evaluation mode").set_value("Post comments").run()
    find_radio(application, "Comment source").set_value("YouTube URL").run()
    application.text_input[0].input("https://youtu.be/AbCdEfGhIjK")
    find_button(application, "Fetch & evaluate").click().run()
    assert not application.exception
    assert message in [item.value for item in application.error]
    assert post.call_count == 1
    assert post.call_args.args == ("http://127.0.0.1:8000/post-comments/youtube",)
    assert post.call_args.kwargs["json"]["max_comments"] == 100


def test_youtube_comments_use_validated_bounded_fetch_then_sentiment(sentiment_dashboard):
    application, _, post = sentiment_dashboard
    fetched = {"comments": ["First", "Second"], "source": "youtube", "scope": "retrieved_top_level_comments",
               "video_id": "AbCdEfGhIjK", "retrieved_count": 2, "received_count": 2,
               "skipped_blank": 0, "skipped_oversized": 0, "has_more": True}
    post.side_effect = [Response(fetched), Response(sentiment_batch(["positive", "negative"]))]
    find_radio(application, "Evaluation mode").set_value("Post comments").run()
    find_radio(application, "Comment source").set_value("YouTube URL").run()
    application.text_input[0].input("https://youtu.be/AbCdEfGhIjK")
    find_button(application, "Fetch & evaluate").click().run()
    assert not application.exception
    assert [call.args[0] for call in post.call_args_list] == ["http://127.0.0.1:8000/post-comments/youtube", "http://127.0.0.1:8000/sentiment"]
    assert [metric.value for metric in application.metric] == ["50.0%", "0.0%", "50.0%"]
    assert any("top-level" in item.value for item in application.caption)


def test_results_are_hidden_when_mode_or_post_source_changes(sentiment_dashboard):
    application, _, post = sentiment_dashboard
    application.text_area[0].input("A helpful explanation.")
    find_button(application, "Evaluate comment").click().run()
    assert not application.exception
    assert len(application.json) == 1
    assert any("Positive sentiment" in item.value for item in application.markdown)

    find_radio(application, "Evaluation mode").set_value("Post comments").run()
    assert not application.exception
    assert not application.json and not application.metric
    assert not any("Positive sentiment" in item.value for item in application.markdown)

    post.return_value = Response(sentiment_batch(["positive", "negative"]))
    application.text_area[0].input("First comment\nSecond comment")
    find_button(application, "Evaluate comments").click().run()
    assert not application.exception
    assert len(application.metric) == 3 and len(application.json) == 1

    for source in ("CSV upload", "YouTube URL"):
        find_radio(application, "Comment source").set_value(source).run()
        assert not application.exception
        assert not application.metric and not application.json and not application.dataframe
    assert post.call_count == 2
