"""Offline tests of optional data and shared-BERT training, using artificial fixtures.

These fixtures are not a project evaluation dataset and establish no accuracy.
Install requirements-training.txt to run this optional group.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="Optional training tests require requirements-training.txt")

from src.config import POSITIVE_LABELS, TOXIC_LABELS
from src.etl import load_goemotions, load_jigsaw, load_twitter_emotion
from src.utils import compute_pos_weight, split_indices_by_source, training_class_weights


def write_csv(tmp_path, labels, rows=None, text_column="text", name="data.csv"):
    if rows is None:
        rows = [{text_column: "A sample comment", **{label: 1 for label in labels}}]
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("loader,labels", [(load_jigsaw, TOXIC_LABELS), (load_goemotions, POSITIVE_LABELS)])
def test_legacy_csv_accepts_complete_binary_schema(tmp_path, loader, labels):
    path = write_csv(tmp_path, labels, text_column="comment_text")
    actual = loader(path)
    assert list(actual.columns) == ["text", *labels]
    assert actual[labels].to_numpy().tolist() == [[1] * len(labels)]


def test_one_label_multilingual_jigsaw_is_rejected(tmp_path):
    path = write_csv(tmp_path, ["toxic"])
    with pytest.raises(ValueError, match="six-label toxicity.*missing columns"):
        load_jigsaw(path)


def test_missing_emotion_is_not_silently_made_negative(tmp_path):
    path = write_csv(tmp_path, POSITIVE_LABELS[:-1])
    with pytest.raises(ValueError, match="missing columns: optimism"):
        load_goemotions(path)


@pytest.mark.parametrize("bad_label", [None, "yes", -1, 2, 0.75])
def test_invalid_binary_label_rejected(tmp_path, bad_label):
    row = {"text": "fixture only", **{label: 0 for label in TOXIC_LABELS}}
    row["toxic"] = bad_label
    with pytest.raises(ValueError, match="only binary 0/1"):
        load_jigsaw(write_csv(tmp_path, TOXIC_LABELS, [row]))


@pytest.mark.parametrize("bad_text", [None, "", "   ", 123])
def test_missing_blank_or_numeric_comments_rejected(tmp_path, bad_text):
    row = {"text": bad_text, **{label: 0 for label in POSITIVE_LABELS}}
    with pytest.raises(ValueError, match="non-text, or blank"):
        load_goemotions(write_csv(tmp_path, POSITIVE_LABELS, [row]))


def test_missing_and_malformed_csv_errors_are_helpful(tmp_path):
    with pytest.raises(ValueError, match="CSV is missing"):
        load_jigsaw(tmp_path / "absent.csv")
    path = tmp_path / "bad.csv"
    path.write_text('text,toxic\n"unclosed', encoding="utf-8")
    with pytest.raises(ValueError, match="empty or malformed"):
        load_jigsaw(path)


def test_conflicting_duplicate_comments_rejected(tmp_path):
    rows = [
        {"text": "Same comment", **{label: 0 for label in TOXIC_LABELS}},
        {"text": "  same   COMMENT  ", **{label: 1 for label in TOXIC_LABELS}},
    ]
    with pytest.raises(ValueError, match="conflicting labels"):
        load_jigsaw(write_csv(tmp_path, TOXIC_LABELS, rows))


def test_twitter_surprise_is_not_invented_as_optimism(tmp_path):
    path = tmp_path / "twitter.csv"
    pd.DataFrame([{"text": "An unexpected turn", "emotion": "surprise"}]).to_csv(path, index=False)
    actual = load_twitter_emotion(path)
    assert actual.empty
    assert list(actual.columns) == ["text", *POSITIVE_LABELS]


def test_split_is_reproducible_and_duplicate_comments_do_not_leak_across_tasks():
    frame = pd.DataFrame([
        {"text": f"Comment {i}", "__src__": "jigsaw"} for i in range(20)
    ] + [
        {"text": f"  COMMENT  {i}  ", "__src__": "goemotions"} for i in range(20)
    ])
    training, validation = split_indices_by_source(frame, 0.2, 42)
    assert (training, validation) == split_indices_by_source(frame, 0.2, 42)
    assert set(training).isdisjoint(validation)
    assert set(training + validation) == set(range(len(frame)))
    normalized = frame.text.map(lambda value: " ".join(value.split()).casefold())
    assert set(normalized.iloc[training]).isdisjoint(normalized.iloc[validation])
    for indices in (training, validation):
        assert set(frame.iloc[indices]["__src__"]) == {"jigsaw", "goemotions"}


def test_small_source_cannot_silently_lose_training_or_validation():
    frame = pd.DataFrame([{"text": "one", "__src__": "jigsaw"}])
    with pytest.raises(ValueError, match="two distinct"):
        split_indices_by_source(frame)


def test_training_weights_do_not_use_holdout_rows():
    frame = pd.DataFrame({"__src__": ["jigsaw"] * 5, "toxic": [1, 0, 0, 1, 1]})
    before = training_class_weights(frame, [0, 1, 2], "jigsaw", ["toxic"])
    frame.loc[[3, 4], "toxic"] = 0
    after = training_class_weights(frame, [0, 1, 2], "jigsaw", ["toxic"])
    np.testing.assert_array_equal(before, [2])
    np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("labels", [np.zeros((2, 1)), np.ones((2, 1)), [[0.5], [1]], np.empty((0, 1))])
def test_unsupported_training_class_distribution_fails(labels):
    with pytest.raises(ValueError):
        compute_pos_weight(np.asarray(labels))


def test_importing_helpers_does_not_initialize_database():
    command = [sys.executable, "-c", "import sys; import src.utils as u; assert u._engine is None; assert 'sqlalchemy' not in sys.modules"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_etl_database_failure_does_not_echo_credentials(monkeypatch, capsys):
    from src import etl

    monkeypatch.setattr(etl, "load_goemotions", lambda _: pd.DataFrame([{"text": "fixture", **{label: 0 for label in POSITIVE_LABELS}}]))
    monkeypatch.setattr(etl, "load_jigsaw", lambda _: pd.DataFrame([{"text": "fixture", **{label: 0 for label in TOXIC_LABELS}}]))
    monkeypatch.setattr(etl, "TWITTER_EMOTION_CSV", "")

    def failing_database(*args, **kwargs):
        raise RuntimeError("postgresql://user:do-not-echo@private/database")

    monkeypatch.setattr(etl, "upsert_documents", failing_database)
    with pytest.raises(RuntimeError, match="Postgres import failed") as failure:
        etl.main()
    assert "do-not-echo" not in str(failure.value)
    assert "do-not-echo" not in capsys.readouterr().out
    assert failure.value.__suppress_context__ is True


def test_missing_training_data_fails_before_network_loading(tmp_path, monkeypatch):
    from src import train_multitask as trainer

    def should_not_download(*args, **kwargs):
        raise AssertionError("No tokenizer/model download is allowed before data validation")

    monkeypatch.setattr(trainer.AutoTokenizer, "from_pretrained", should_not_download)
    with pytest.raises(ValueError, match="CSV is missing"):
        trainer.train(jigsaw_csv=str(tmp_path / "missing.csv"), goemotions_csv=str(tmp_path / "missing2.csv"))


def test_unknown_task_does_not_train_its_head():
    import torch
    from src.train_multitask import masked_multitask_loss

    toxicity = torch.zeros((1, 6), requires_grad=True)
    positivity = torch.zeros((1, 6), requires_grad=True)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss = masked_multitask_loss(
        toxicity, positivity, torch.ones((1, 6)), torch.full((1, 6), float("nan")), criterion, criterion,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(toxicity.grad) == 6
    assert torch.count_nonzero(positivity.grad) == 0


def test_per_label_metrics_are_measured_with_support_and_zero_denominators():
    import torch
    from src.train_multitask import classification_metrics

    actual = classification_metrics(torch.tensor([[1, 0], [1, 0]]), torch.tensor([[1, 0], [0, 0]]), ["known", "absent"])
    assert actual["n_examples"] == 2
    assert actual["per_label"]["known"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "support": 2}
    assert actual["per_label"]["absent"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def test_tiny_offline_bert_trains_selects_checkpoint_and_loads_for_inference(tmp_path, monkeypatch):
    """Exercise the full training loop with tiny random BERT and artificial CSVs."""
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from src import train_multitask as trainer

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "this", "is", "sample", "good", "bad", *map(str, range(12))]
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary))
    tiny_config = BertConfig(
        vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, max_position_embeddings=32, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    original_model_class = trainer.CivicBalance
    captured = {}

    def tiny_model(*args, **kwargs):
        model = original_model_class(backbone=BertModel(tiny_config))
        captured["initial_weights"] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        return model

    monkeypatch.setattr(trainer, "CivicBalance", tiny_model)
    monkeypatch.setattr(trainer.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    paths = {}
    for source, labels in (("jigsaw", TOXIC_LABELS), ("goemotions", POSITIVE_LABELS)):
        rows = [{"text": f"this is sample {i} {'good' if i % 2 else 'bad'}", **{label: i % 2 for label in labels}} for i in range(12)]
        paths[source] = write_csv(tmp_path, labels, rows, name=f"{source}.csv")
    output = tmp_path / "checkpoint"
    metadata = trainer.train(
        jigsaw_csv=str(paths["jigsaw"]), goemotions_csv=str(paths["goemotions"]), output_dir=str(output),
        max_len=8, epochs=2, batch_size=4, lr=0.01, seed=42, device="cpu", validation_fraction=0.25,
    )
    assert metadata["best_epoch"] in {1, 2}
    assert metadata["split_counts"] == {"training": {"jigsaw": 9, "goemotions": 9}, "validation": {"jigsaw": 3, "goemotions": 3}}
    assert len(json.loads((output / "training_history.json").read_text())) == 2
    assert (output / "model.safetensors").exists()
    assert (output / "config.json").exists()
    assert (output / "tokenizer.json").exists()
    assert "this is sample" not in (output / "training_metadata.json").read_text()
    monkeypatch.undo()
    # This local reload cannot download anything: loader uses local_files_only.
    loaded_tokenizer, loaded_model, loaded_metadata = trainer.load_local_checkpoint(output)
    assert loaded_metadata["max_tokens"] == 8
    assert any(not torch.equal(value, captured["initial_weights"][key]) for key, value in loaded_model.state_dict().items())
    with torch.inference_mode():
        toxicity, positivity = loaded_model(**loaded_tokenizer("this is good", return_tensors="pt"))
    assert toxicity.shape == positivity.shape == (1, 6)
    assert torch.isfinite(toxicity).all() and torch.isfinite(positivity).all()
    from src.inference import CommentEvaluator
    evaluator = CommentEvaluator(backend="shared_bert", artifact_dir=output)
    evaluator.load()
    prediction = evaluator.predict(["this is good"])[0]
    assert set(prediction["toxicity"]) == set(TOXIC_LABELS)
    assert set(prediction["positivity"]) == set(POSITIVE_LABELS)
    assert isinstance(prediction["label"], str)
    assert "confidence" not in prediction


def test_shared_checkpoint_bad_metadata_fails_before_any_model_load(tmp_path):
    from src.train_multitask import load_local_checkpoint

    (tmp_path / "training_metadata.json").write_text(json.dumps({"artifact_format": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_local_checkpoint(tmp_path)
