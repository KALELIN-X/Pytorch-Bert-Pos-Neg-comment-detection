"""Small, explicit request and response contracts shared by the API tests."""

import math
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictStr, model_validator

from .config import MAX_BATCH_COMMENTS, MAX_COMMENT_CHARS, POSITIVE_LABELS, TOXIC_LABELS

UnitScore = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Weight = Annotated[float, Field(strict=True, ge=0, le=3, allow_inf_nan=False)]
BinaryFlag = Annotated[int, Field(strict=True, ge=0, le=1)]


def _strip_comment(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Each comment must contain non-whitespace text.")
    return value


# The raw length is bounded before trimming; a batch error retains its item index.
CommentText = Annotated[StrictStr, Field(min_length=1, max_length=MAX_COMMENT_CHARS), AfterValidator(_strip_comment)]


class CommentInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [
            {"text": "Thank you for helping me understand this."},
            {"texts": ["Thank you for your help.", "I disagree with this idea."]},
        ],
    })

    text: CommentText | None = None
    texts: Annotated[list[CommentText], Field(min_length=1, max_length=MAX_BATCH_COMMENTS)] | None = None
    @model_validator(mode="before")
    @classmethod
    def require_one_input(cls, value):
        if isinstance(value, dict):
            present = [name for name in ("text", "texts") if name in value]
            if len(present) != 1:
                raise ValueError("Provide exactly one of 'text' or 'texts'.")
            if value[present[0]] is None:
                raise ValueError("Comment input cannot be null.")
        return value

    def comments(self) -> list[str]:
        if self.texts is not None:
            return self.texts
        assert self.text is not None  # Guaranteed by require_one_input.
        return [self.text]


class PredictionRequest(CommentInputs):
    threshold: UnitScore = Field(default=0.5, description="Inclusive activation threshold, 0–1. At 0 every label is active; at 1 only scores of 1 are active.")
    w_pos: Weight = Field(default=1.0, description="Weight of average positive-emotion score in the net-goodness heuristic, 0–3.")
    w_tox: Weight = Field(default=1.2, description="Weight of average toxicity score subtracted in the net-goodness heuristic, 0–3.")


class SentimentRequest(CommentInputs):
    """Sentence sentiment has no emotion/toxicity threshold or heuristic weights."""


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toxicity: dict[str, UnitScore]
    positivity: dict[str, UnitScore]
    toxicity_bin: dict[str, BinaryFlag]
    positivity_bin: dict[str, BinaryFlag]
    net_goodness: Annotated[float, Field(strict=True, ge=-3, le=3, allow_inf_nan=False)]
    label: Literal["positive_signals", "toxicity_signals", "mixed_signals", "no_detected_signals"]
    explanation: Annotated[StrictStr, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_label_sets(self):
        for scores, expected in (
            (self.toxicity, TOXIC_LABELS), (self.toxicity_bin, TOXIC_LABELS),
            (self.positivity, POSITIVE_LABELS), (self.positivity_bin, POSITIVE_LABELS),
        ):
            if set(scores) != set(expected):
                raise ValueError("Model output must contain exactly the expected labels.")
        return self


class ScoreMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_type: Literal["uncalibrated_model_scores"]
    confidence_calibrated: Literal[False]
    description: StrictStr
    backend: Literal["public", "shared_bert", "unconfigured"]
    max_tokens: Annotated[int, Field(strict=True, ge=3, le=192)]
    threshold: UnitScore
    w_pos: Weight
    w_tox: Weight


class SinglePredictionResponse(PredictionResult):
    metadata: ScoreMetadata


class IndexedPredictionResult(PredictionResult):
    index: Annotated[int, Field(strict=True, ge=0, lt=MAX_BATCH_COMMENTS)]


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: Annotated[list[IndexedPredictionResult], Field(min_length=1, max_length=MAX_BATCH_COMMENTS)]
    count: Annotated[int, Field(strict=True, ge=1, le=MAX_BATCH_COMMENTS)]
    metadata: ScoreMetadata


SentimentLabel = Literal["positive", "neutral", "negative"]
SENTIMENT_LABELS = ("positive", "neutral", "negative")


class SentimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: SentimentLabel
    scores: dict[str, UnitScore]
    explanation: Annotated[StrictStr, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_scores(self):
        if set(self.scores) != set(SENTIMENT_LABELS):
            raise ValueError("Sentiment scores must contain positive, neutral and negative.")
        if not math.isclose(sum(self.scores.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Sentiment softmax scores must sum to one.")
        if self.scores[self.label] != max(self.scores.values()):
            raise ValueError("The predicted sentiment must have a maximum model score.")
        return self


class SentimentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_type: Literal["uncalibrated_model_scores"]
    confidence_calibrated: Literal[False]
    description: StrictStr
    max_tokens: Annotated[int, Field(strict=True, ge=3, le=192)]


class SingleSentimentResponse(SentimentResult):
    metadata: SentimentMetadata


class IndexedSentimentResult(SentimentResult):
    index: Annotated[int, Field(strict=True, ge=0, lt=MAX_BATCH_COMMENTS)]


class SentimentAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total: Annotated[int, Field(strict=True, ge=1, le=MAX_BATCH_COMMENTS)]
    counts: dict[str, Annotated[int, Field(strict=True, ge=0, le=MAX_BATCH_COMMENTS)]]
    percentages: dict[str, Annotated[float, Field(strict=True, ge=0, le=100, allow_inf_nan=False)]]

    @model_validator(mode="after")
    def validate_aggregate(self):
        if set(self.counts) != set(SENTIMENT_LABELS) or set(self.percentages) != set(SENTIMENT_LABELS):
            raise ValueError("Aggregate keys must contain exactly the three sentiment labels.")
        if sum(self.counts.values()) != self.total:
            raise ValueError("Aggregate counts must sum to the analyzed total.")
        if any(not math.isclose(self.percentages[label], count / self.total * 100, abs_tol=1e-6)
               for label, count in self.counts.items()):
            raise ValueError("Percentages must represent predicted-label counts divided by the analyzed total.")
        return self


class BatchSentimentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: Annotated[list[IndexedSentimentResult], Field(min_length=1, max_length=MAX_BATCH_COMMENTS)]
    count: Annotated[int, Field(strict=True, ge=1, le=MAX_BATCH_COMMENTS)]
    aggregate: SentimentAggregate
    metadata: SentimentMetadata

    @model_validator(mode="after")
    def validate_results(self):
        if self.count != len(self.results) or self.aggregate.total != self.count:
            raise ValueError("Batch counts must match the analyzed results.")
        if [result.index for result in self.results] != list(range(self.count)):
            raise ValueError("Batch results must have ordered, contiguous indexes.")
        if any(self.aggregate.counts[label] != sum(result.label == label for result in self.results)
               for label in SENTIMENT_LABELS):
            raise ValueError("Aggregate counts must represent the actual predicted labels.")
        return self


class YouTubeCommentsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: Annotated[StrictStr, Field(min_length=1, max_length=2048), AfterValidator(_strip_comment)]
    max_comments: Annotated[int, Field(strict=True, ge=1, le=100)] = 100


class YouTubeCommentsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comments: Annotated[list[CommentText], Field(max_length=100)]
    source: Literal["youtube"]
    scope: Literal["retrieved_top_level_comments"]
    retrieved_count: Annotated[int, Field(strict=True, ge=0, le=100)]
    received_count: Annotated[int, Field(strict=True, ge=0, le=100)]
    skipped_blank: Annotated[int, Field(strict=True, ge=0, le=100)]
    skipped_oversized: Annotated[int, Field(strict=True, ge=0, le=100)]
    has_more: bool
    video_id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9_-]{11}$")]

    @model_validator(mode="after")
    def validate_count(self):
        if self.retrieved_count != len(self.comments):
            raise ValueError("Retrieved count must match the returned comments.")
        if self.received_count != self.retrieved_count + self.skipped_blank + self.skipped_oversized:
            raise ValueError("Received count must include returned and skipped comments.")
        return self
