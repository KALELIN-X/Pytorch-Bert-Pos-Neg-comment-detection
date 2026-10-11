"""Controlled-output tests for newly added sentiment; not dataset evaluation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.inference import InputTooLongError, ModelUnavailableError, PredictionError
from src.sentiment import LABELS, SentimentEvaluator, _label_mapping, read_manifest


class Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(kwargs)
        lengths = [len(text.split()) + 2 for text in texts]
        mask = torch.tensor([[1] * size + [0] * (max(lengths) - size) for size in lengths])
        return {"input_ids": mask, "attention_mask": mask, "token_type_ids": torch.zeros_like(mask)}


class Model:
    def __init__(self, logits=None, labels=None, error=None):
        self.logits = torch.tensor([[3., 0., -1.]]) if logits is None else logits
        self.config = SimpleNamespace(num_labels=3, id2label=dict(enumerate(labels or LABELS)))
        self.error = error
        self.calls = 0

    def __call__(self, **encoding):
        self.calls += 1
        assert "token_type_ids" not in encoding  # DistilBERT has no segment embeddings.
        if self.error:
            raise RuntimeError(self.error)
        return SimpleNamespace(logits=self.logits)

    def to(self, device):
        assert device == "cpu"
        return self

    def eval(self):
        return self


def loaded_engine(logits=None, labels=None):
    engine = SentimentEvaluator()
    engine._tokenizer = Tokenizer()
    engine._model = Model(logits, labels)
    engine._labels = dict(enumerate(labels or LABELS))
    return engine


def test_one_argmax_label_per_comment_preserves_order_and_normalizes_scores():
    logits = torch.tensor([[4., 0., -2.], [-1., 0., 4.], [0., 5., 0.]])
    engine = loaded_engine(logits)
    results = engine.predict(["Illustrative positive", "Illustrative negative", "Illustrative neutral"])
    assert [result["label"] for result in results] == ["positive", "negative", "neutral"]
    for result in results:
        assert set(result["scores"]) == set(LABELS)
        assert sum(result["scores"].values()) == pytest.approx(1)
        assert result["label"] == max(result["scores"], key=result["scores"].get)
        assert "uncalibrated" in result["explanation"]
        assert "text" not in result
        assert "confidence" not in result


def test_label_selection_maps_model_indices_by_name():
    engine = loaded_engine(torch.tensor([[0., 4., -2.]]), ["negative", "positive", "neutral"])
    result = engine.predict(["A sample"])[0]
    assert result["label"] == "positive"
    assert result["scores"]["positive"] > 0.9


@pytest.mark.parametrize("texts", ["comment", [], [None], [7], [""], [" \t\n"], ["x" * 2001], ["x"] * 21])
def test_invalid_inputs_are_rejected_before_model_work(texts):
    engine = loaded_engine()
    with pytest.raises(ValueError):
        engine.predict(texts)
    assert engine._model.calls == 0
    assert engine._tokenizer.calls == []


def test_prediction_does_not_implicitly_load_or_download():
    with pytest.raises(ModelUnavailableError, match="not loaded"):
        SentimentEvaluator().predict(["A sample"])


def test_token_limit_rejects_correct_batch_item_without_truncation():
    engine = loaded_engine(torch.zeros((2, 3)))
    with pytest.raises(InputTooLongError) as failure:
        engine.predict(["Short sentence", "word " * 191])
    assert failure.value.index == 1
    assert failure.value.max_tokens == 192
    assert engine._model.calls == 0
    assert engine._tokenizer.calls[0]["truncation"] is False


def test_exact_token_budget_is_accepted_including_special_tokens():
    engine = loaded_engine()
    assert engine.predict(["word " * 190])[0]["label"] == "positive"
    assert engine._model.calls == 1


@pytest.mark.parametrize("logits", [
    torch.tensor([[float("nan"), 0, 1]]), torch.tensor([[float("inf"), 0, 1]]),
    torch.zeros((1, 4)), torch.zeros((2, 3)), torch.zeros((3,)),
    torch.zeros((1, 3), dtype=torch.int64), [1., 2., 3.],
])
def test_malformed_predictions_fail_clearly(logits):
    with pytest.raises(PredictionError, match="invalid prediction values"):
        loaded_engine(logits).predict(["A sample"])


def test_raw_prediction_error_does_not_expose_input_or_credentials():
    engine = loaded_engine()
    engine._model.error = "credential=synthetic-secret private-input"
    with pytest.raises(PredictionError) as failure:
        engine.predict(["private-input"])
    assert str(failure.value).startswith("Sentiment prediction failed.")
    assert "synthetic-secret" not in str(failure.value)
    assert "private-input" not in str(failure.value)


@pytest.mark.parametrize("mapping", [
    {0: "positive", 1: "neutral", 2: "unrelated"},
    {0: "positive", 1: "positive", 2: "negative"},
    {1: "positive", 2: "neutral", 3: "negative"},
])
def test_incompatible_label_mapping_is_rejected(mapping):
    config = SimpleNamespace(num_labels=3, id2label=mapping)
    with pytest.raises(ModelUnavailableError, match="incompatible labels"):
        _label_mapping(config)


def make_artifacts(folder):
    manifest = read_manifest()
    for name in manifest["required_files"]:
        (folder / name).write_bytes(b"synthetic fixture")
    (folder / "download_manifest.json").write_text(json.dumps({
        "repo_id": manifest["repo_id"], "revision": manifest["revision"],
    }), encoding="utf-8")


def fake_loaders(monkeypatch, *, fail=False):
    import transformers
    calls = []

    def tokenizer(path, **kwargs):
        calls.append(("tokenizer", Path(path), kwargs))
        return Tokenizer()

    def model(path, **kwargs):
        calls.append(("model", Path(path), kwargs))
        if fail:
            raise RuntimeError("private input and credential=synthetic-secret")
        return Model()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", model)
    return calls


def test_missing_artifacts_do_not_attempt_network_loading(monkeypatch, tmp_path):
    calls = fake_loaders(monkeypatch)
    engine = SentimentEvaluator(model_dir=tmp_path)
    with pytest.raises(ModelUnavailableError, match="artifacts are missing"):
        engine.load()
    assert not engine.ready
    assert calls == []


def test_local_loader_uses_safetensors_no_remote_code_and_loads_once(monkeypatch, tmp_path):
    make_artifacts(tmp_path)
    calls = fake_loaders(monkeypatch)
    engine = SentimentEvaluator(model_dir=tmp_path)
    engine.load()
    engine.load()
    assert engine.ready
    assert len(calls) == 2
    for kind, folder, kwargs in calls:
        assert folder == tmp_path
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        if kind == "model":
            assert kwargs["use_safetensors"] is True
    assert engine.predict(["A sample"])[0]["label"] == "positive"


def test_wrong_provenance_is_rejected_before_loading(monkeypatch, tmp_path):
    make_artifacts(tmp_path)
    (tmp_path / "download_manifest.json").write_text("{}", encoding="utf-8")
    calls = fake_loaders(monkeypatch)
    with pytest.raises(ModelUnavailableError, match="provenance"):
        SentimentEvaluator(model_dir=tmp_path).load()
    assert calls == []


def test_loading_failure_stays_unready_and_does_not_expose_raw_error(monkeypatch, tmp_path):
    make_artifacts(tmp_path)
    fake_loaders(monkeypatch, fail=True)
    engine = SentimentEvaluator(model_dir=tmp_path)
    with pytest.raises(ModelUnavailableError) as failure:
        engine.load()
    assert not engine.ready
    assert "synthetic-secret" not in str(failure.value)


def test_downloader_is_pinned_anonymous_and_accepts_only_known_files(monkeypatch, tmp_path):
    from src import download_sentiment
    calls = []
    manifest = read_manifest()

    def download(**kwargs):
        calls.append(kwargs)
        for name in manifest["required_files"]:
            (kwargs["local_dir"] / name).write_bytes(b"synthetic fixture")

    monkeypatch.setattr(download_sentiment, "snapshot_download", download)
    download_sentiment.download_sentiment(tmp_path)
    assert len(calls) == 1
    assert calls[0]["repo_id"] == manifest["repo_id"]
    assert calls[0]["revision"] == manifest["revision"]
    assert calls[0]["token"] is False
    assert calls[0]["allow_patterns"] == manifest["allow_patterns"]
    assert not any(".bin" in name or ".py" in name for name in manifest["allow_patterns"])
    assert json.loads((tmp_path / "download_manifest.json").read_text(encoding="utf-8")) == {
        "repo_id": manifest["repo_id"], "revision": manifest["revision"],
    }


def test_partial_download_never_marks_model_ready(monkeypatch, tmp_path):
    from src import download_sentiment
    monkeypatch.setattr(download_sentiment, "snapshot_download", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="incomplete"):
        download_sentiment.download_sentiment(tmp_path)
    assert not (tmp_path / "download_manifest.json").exists()
