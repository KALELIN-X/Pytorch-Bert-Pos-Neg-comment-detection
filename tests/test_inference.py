"""Behavioral tests use controlled model outputs, not claimed accuracy scores."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from src.config import POSITIVE_LABELS, TOXIC_LABELS
from src.inference import (
    CommentEvaluator,
    InputTooLongError,
    ModelUnavailableError,
    PredictionError,
    _LoadedModel,
    _label_indices,
    read_manifest,
)


class FakeTokenizer:
    def __init__(self, extra_tokens=2):
        self.extra_tokens = extra_tokens
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(kwargs)
        lengths = [len(text.split()) + self.extra_tokens for text in texts]
        mask = torch.tensor([[1] * length + [0] * (max(lengths) - length) for length in lengths])
        return {"input_ids": mask, "attention_mask": mask}


class FakeModel:
    def __init__(self, scores=None, *, logits=None, config=None, error=None):
        self.logits = torch.logit(torch.tensor(scores, dtype=torch.float32)) if logits is None else logits
        self.config = config
        self.error = error
        self.calls = 0
        self.eval_called = False

    def __call__(self, **encoding):
        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        return SimpleNamespace(logits=self.logits)

    def to(self, device):
        assert device == "cpu"
        return self

    def eval(self):
        self.eval_called = True
        return self


def evaluator_with_scores(tox, pos, *, pos_extra_tokens=2):
    engine = CommentEvaluator()
    engine._models = {
        "toxicity": _LoadedModel(FakeTokenizer(), FakeModel(tox), {label: i for i, label in enumerate(TOXIC_LABELS)}, 6),
        "positive_emotions": _LoadedModel(FakeTokenizer(pos_extra_tokens), FakeModel(pos), {label: i for i, label in enumerate(POSITIVE_LABELS)}, 6),
    }
    return engine


def test_batch_preserves_order_original_categories_and_weighted_formula():
    engine = evaluator_with_scores([[0.1] * 6, [0.9] * 6], [[0.8] * 6, [0.1] * 6])
    results = engine.predict(["Thank you for helping", "You are awful"], w_pos=2, w_tox=1.2)
    assert [result["label"] for result in results] == ["positive_signals", "toxicity_signals"]
    assert results[0]["net_goodness"] == pytest.approx(2 * 0.8 - 1.2 * 0.1)
    assert results[1]["net_goodness"] == pytest.approx(2 * 0.1 - 1.2 * 0.9)
    assert list(results[0]["toxicity"]) == TOXIC_LABELS
    assert list(results[0]["positivity"]) == POSITIVE_LABELS
    assert all(value == 1 for value in results[0]["positivity_bin"].values())
    assert all(value == 0 for value in results[0]["toxicity_bin"].values())
    assert "text" not in results[0]
    assert "confidence" not in results[0]


@pytest.mark.parametrize("positive,toxic,label", [
    (0.8, 0.1, "positive_signals"),
    (0.1, 0.8, "toxicity_signals"),
    (0.8, 0.8, "mixed_signals"),
    (0.1, 0.1, "no_detected_signals"),
])
def test_summary_describes_detected_signals_without_assuming_sentiment(positive, toxic, label):
    engine = evaluator_with_scores([[toxic] * 6], [[positive] * 6])
    result = engine.predict(["A demonstration comment"])[0]
    assert result["label"] == label
    assert result["explanation"]
    if label == "no_detected_signals":
        assert "does not establish neutral or negative sentiment" in result["explanation"]


def test_inclusive_threshold_and_weights_change_only_the_heuristic():
    engine = evaluator_with_scores([[0.5] * 6], [[0.5] * 6])
    result = engine.predict(["A comment"], threshold=0.5, w_pos=0, w_tox=0)[0]
    assert result["label"] == "mixed_signals"
    assert result["net_goodness"] == 0
    result = engine.predict(["A comment"], threshold=0.6)[0]
    assert result["label"] == "no_detected_signals"


@pytest.mark.parametrize("texts", ["comment", [], [None], [12], [""], [" \n\t"], ["x" * 2001], ["x"] * 21])
def test_invalid_comment_inputs_are_rejected_before_models_run(texts):
    engine = evaluator_with_scores([[0.1] * 6], [[0.1] * 6])
    with pytest.raises(ValueError):
        engine.predict(texts)
    assert all(loaded.model.calls == 0 for loaded in engine._models.values())


@pytest.mark.parametrize("parameters", [
    {"threshold": float("nan")}, {"threshold": float("inf")}, {"threshold": -0.1},
    {"threshold": 1.1}, {"threshold": True}, {"w_pos": -1}, {"w_tox": 3.1},
    {"w_pos": float("inf")}, {"w_tox": "1"},
])
def test_invalid_thresholds_and_weights_are_rejected(parameters):
    with pytest.raises(ValueError):
        CommentEvaluator().predict(["A comment"], **parameters)


def test_predict_does_not_implicitly_load_or_download():
    with pytest.raises(ModelUnavailableError, match="Models are not loaded"):
        CommentEvaluator().predict(["A comment"])


def test_token_budget_identifies_batch_item_and_never_truncates():
    engine = evaluator_with_scores([[0.1] * 6] * 2, [[0.1] * 6] * 2)
    with pytest.raises(InputTooLongError) as failure:
        engine.predict(["Short comment", "word " * 191])
    assert failure.value.index == 1
    assert failure.value.max_tokens == 192
    assert all(loaded.model.calls == 0 for loaded in engine._models.values())
    assert engine._models["toxicity"].tokenizer.calls[0]["truncation"] is False


def test_both_tokenizers_are_validated_before_either_model_runs():
    engine = evaluator_with_scores([[0.1] * 6], [[0.1] * 6], pos_extra_tokens=193)
    with pytest.raises(InputTooLongError):
        engine.predict(["Short"])
    assert all(loaded.model.calls == 0 for loaded in engine._models.values())


@pytest.mark.parametrize("logits", [
    torch.tensor([[float("nan")] * 6]), torch.tensor([[float("inf")] * 6]),
    torch.zeros((1, 5)), torch.zeros((2, 6)), torch.zeros((6,)),
    torch.zeros((1, 6), dtype=torch.int64), [0.1] * 6,
])
def test_invalid_prediction_outputs_fail_clearly(logits):
    engine = evaluator_with_scores([[0.1] * 6], [[0.1] * 6])
    engine._models["toxicity"].model.logits = logits
    with pytest.raises(PredictionError, match="invalid prediction values"):
        engine.predict(["A comment"])


def test_prediction_failure_never_exposes_comments_or_raw_exception():
    private_text = "private input should never be returned"
    engine = evaluator_with_scores([[0.1] * 6], [[0.1] * 6])
    engine._models["toxicity"].model.error = "credential=secret-value " + private_text
    with pytest.raises(PredictionError) as failure:
        engine.predict([private_text])
    assert str(failure.value).startswith("Prediction failed.")
    assert "secret-value" not in str(failure.value)
    assert private_text not in str(failure.value)


def config_for(labels):
    return SimpleNamespace(
        num_labels=len(labels), id2label={str(i): label for i, label in enumerate(labels)},
        problem_type="multi_label_classification",
    )


def test_label_mapping_uses_names_instead_of_position():
    reversed_labels = list(reversed(TOXIC_LABELS))
    indices = _label_indices(config_for(reversed_labels), TOXIC_LABELS, 6)
    assert indices["toxic"] == 5
    assert indices["identity_hate"] == 0


@pytest.mark.parametrize("change", ["missing_label", "duplicate_label", "wrong_count", "wrong_problem"])
def test_incompatible_artifact_label_mapping_is_rejected(change):
    config = config_for(TOXIC_LABELS)
    if change == "missing_label":
        config.id2label["0"] = "unrelated"
    elif change == "duplicate_label":
        config.id2label["0"] = "threat"
    elif change == "wrong_count":
        config.num_labels = 28
    else:
        config.problem_type = "single_label_classification"
    with pytest.raises(ModelUnavailableError, match="incompatible"):
        _label_indices(config, TOXIC_LABELS, 6)


def prepare_artifacts(directory):
    manifest = read_manifest()
    provenance = {}
    for spec in manifest["models"]:
        folder = directory / spec["key"]
        folder.mkdir(parents=True)
        for filename in spec["required_files"]:
            (folder / filename).write_bytes(b"test placeholder")
        provenance[spec["key"]] = {"repo_id": spec["repo_id"], "revision": spec["revision"]}
    (directory / "download_manifest.json").write_text(json.dumps(provenance), encoding="utf-8")


def install_fake_public_loaders(monkeypatch, *, fail_positive=False):
    import transformers
    calls = []

    def tokenizer_loader(folder, **kwargs):
        calls.append(("tokenizer", Path(folder).name, kwargs))
        return FakeTokenizer()

    def model_loader(folder, **kwargs):
        key = Path(folder).name
        calls.append(("model", key, kwargs))
        if key == "positive_emotions" and fail_positive:
            raise RuntimeError("do not show credential=secret-value")
        labels = TOXIC_LABELS if key == "toxicity" else POSITIVE_LABELS + [f"other{i}" for i in range(22)]
        return FakeModel([[0.1] * len(labels)], config=config_for(labels))

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", model_loader)
    return calls


def test_loading_missing_artifacts_fails_without_remote_load(monkeypatch, tmp_path):
    calls = install_fake_public_loaders(monkeypatch)
    engine = CommentEvaluator(model_dir=tmp_path)
    with pytest.raises(ModelUnavailableError, match="models are missing"):
        engine.load()
    assert not engine.ready
    assert calls == []


def test_public_loader_is_local_only_safetensors_and_idempotent(monkeypatch, tmp_path):
    prepare_artifacts(tmp_path)
    calls = install_fake_public_loaders(monkeypatch)
    engine = CommentEvaluator(model_dir=tmp_path)
    engine.load()
    assert engine.ready
    assert len(calls) == 4
    for kind, _, kwargs in calls:
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        if kind == "model":
            assert kwargs["use_safetensors"] is True
    result = engine.predict(["Demonstration"])[0]
    assert set(result["positivity"]) == set(POSITIVE_LABELS)
    engine.load()
    assert len(calls) == 4


def test_loading_failure_does_not_publish_partial_models_or_leak_raw_errors(monkeypatch, tmp_path):
    prepare_artifacts(tmp_path)
    install_fake_public_loaders(monkeypatch, fail_positive=True)
    engine = CommentEvaluator(model_dir=tmp_path)
    with pytest.raises(ModelUnavailableError) as failure:
        engine.load()
    assert not engine.ready
    assert engine._models == {}
    assert "secret-value" not in str(failure.value)


def test_wrong_download_revision_is_rejected_before_loading(monkeypatch, tmp_path):
    prepare_artifacts(tmp_path)
    path = tmp_path / "download_manifest.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["toxicity"]["revision"] = "different-revision"
    path.write_text(json.dumps(provenance), encoding="utf-8")
    calls = install_fake_public_loaders(monkeypatch)
    with pytest.raises(ModelUnavailableError, match="provenance"):
        CommentEvaluator(model_dir=tmp_path).load()
    assert calls == []


def shared_metadata():
    return {
        "artifact_format": "civicbalance-shared-bert-v1", "max_tokens": 128,
        "positive_labels": POSITIVE_LABELS, "toxic_labels": TOXIC_LABELS,
    }


class FakeSharedModel(FakeModel):
    def __call__(self, **encoding):
        self.calls += 1
        return torch.logit(torch.tensor([[0.1] * 6])), torch.logit(torch.tensor([[0.8] * 6]))


def test_shared_bert_backend_uses_local_checkpoint_and_original_dual_heads(monkeypatch, tmp_path):
    model = FakeSharedModel([[0.1]])
    called = []

    def load(folder):
        called.append(folder)
        return FakeTokenizer(), model, shared_metadata()

    monkeypatch.setitem(sys.modules, "src.train_multitask", SimpleNamespace(load_local_checkpoint=load))
    engine = CommentEvaluator(backend="shared_bert", artifact_dir=tmp_path)
    engine.load()
    result = engine.predict(["Thank you for the help"])[0]
    assert result["label"] == "positive_signals"
    assert engine.ready
    assert engine.max_tokens == 128
    assert model.calls == 1
    assert called == [tmp_path]
    assert "train_multitask" in engine.setup_command


def test_shared_checkpoint_incompatible_labels_are_rejected(monkeypatch):
    metadata = shared_metadata()
    metadata["positive_labels"] = list(reversed(POSITIVE_LABELS))
    monkeypatch.setitem(sys.modules, "src.train_multitask", SimpleNamespace(
        load_local_checkpoint=lambda folder: (FakeTokenizer(), FakeSharedModel([[0.1]]), metadata)
    ))
    engine = CommentEvaluator(backend="shared_bert")
    with pytest.raises(ModelUnavailableError, match="metadata"):
        engine.load()
    assert not engine.ready


def test_unknown_backend_fails_during_load_without_crashing_constructor():
    engine = CommentEvaluator(backend="invalid")
    assert not engine.ready
    with pytest.raises(ModelUnavailableError, match="INFERENCE_BACKEND"):
        engine.load()


def test_explicit_download_pins_revisions_files_and_disables_implicit_credentials(monkeypatch, tmp_path):
    from src import download_models
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        folder = kwargs["local_dir"]
        folder.mkdir(parents=True)
        spec = next(spec for spec in read_manifest()["models"] if spec["repo_id"] == kwargs["repo_id"])
        for filename in spec["required_files"]:
            (folder / filename).write_bytes(b"test placeholder")

    monkeypatch.setattr(download_models, "snapshot_download", download)
    download_models.download_models(tmp_path)
    manifest = read_manifest()
    assert len(calls) == 2
    for call, spec in zip(calls, manifest["models"]):
        assert call["repo_id"] == spec["repo_id"]
        assert call["revision"] == spec["revision"]
        assert call["token"] is False
        assert "model.safetensors" in call["allow_patterns"]
        assert not any(".bin" in item or "*.py" == item for item in call["allow_patterns"])
    assert (tmp_path / "download_manifest.json").is_file()


def test_partial_download_does_not_mark_models_ready(monkeypatch, tmp_path):
    from src import download_models

    def incomplete(**kwargs):
        kwargs["local_dir"].mkdir(parents=True)

    monkeypatch.setattr(download_models, "snapshot_download", incomplete)
    with pytest.raises(RuntimeError, match="incomplete"):
        download_models.download_models(tmp_path)
    assert not (tmp_path / "download_manifest.json").exists()
