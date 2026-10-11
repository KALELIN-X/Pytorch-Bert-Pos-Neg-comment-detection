"""Local, bounded comment prediction API. Submitted comments are not persisted."""

from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import API_MAX_TOKENS, INFERENCE_BACKEND, MAX_BATCH_COMMENTS, MAX_COMMENT_CHARS, MAX_REQUEST_BYTES, YOUTUBE_API_KEY
from .inference import CommentEvaluator, InputTooLongError, ModelUnavailableError, PredictionError
from .platform_comments import InvalidPostURLError, MissingPlatformAccessError, PlatformFetchError, fetch_youtube_comments
from .schemas import (
    BatchPredictionResponse, BatchSentimentResponse, PredictionRequest, PredictionResult,
    SENTIMENT_LABELS, SentimentRequest, SentimentResult, SinglePredictionResponse,
    SingleSentimentResponse, YouTubeCommentsRequest, YouTubeCommentsResponse,
)
from .sentiment import SentimentEvaluator

DOWNLOAD_COMMAND = "python -m src.download_models"
SENTIMENT_DOWNLOAD_COMMAND = "python -m src.download_sentiment"
SCORE_DESCRIPTION = (
    "Scores are uncalibrated model outputs, not confidence in the summary label. "
    "Net goodness is a weighted heuristic, not a probability or accuracy measure."
)
LANDING_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CommentScope API</title>
<style>body{margin:0;background:#101722;color:#edf3fa;font:17px/1.6 system-ui,sans-serif}
main{max-width:760px;margin:12vh auto;padding:32px}h1{font-size:42px;line-height:1.15}
p{color:#b8c7d9}a{color:#72d6c1}code{background:#263448;padding:4px 8px;border-radius:6px}
nav{display:flex;gap:24px;flex-wrap:wrap;margin-top:28px}.eyebrow{color:#72d6c1;letter-spacing:.15em}
</style></head><body><main><div class="eyebrow">LOCAL COMMENT EVALUATION</div>
<h1>CommentScope API</h1><p>Use <code>POST /sentiment</code> for positive, neutral or negative
sentence sentiment, with one result per comment. Batch percentages count predicted labels among the analyzed comments.</p>
<p>The preserved <code>POST /predict</code> workflow explores six positive emotions and six toxicity signals.
Its summary describes detected signals, rather than general sentiment. Scores are uncalibrated model outputs,
and net goodness is a weighted heuristic. Comments are not stored by this API.</p>
<p>Optional <code>POST /post-comments/youtube</code> retrieves up to 100 recent top-level comments
through the official YouTube API when a local API key is configured.</p>
<nav><a href="/docs">Try the interactive API</a><a href="/health">Check model readiness</a>
<a href="/redoc">Read the API reference</a></nav></main></body></html>"""


class BodySizeLimitMiddleware:
    """Read at most one bounded body, including requests without Content-Length."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = -1
            if declared_size < 0:
                await JSONResponse({"detail": "Invalid Content-Length header."}, status_code=400)(scope, receive, send)
                return
            if declared_size > MAX_REQUEST_BYTES:
                await self.reject(scope, receive, send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                await self.reject(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        async def bounded_receive():
            nonlocal body
            if body is not None:
                message = {"type": "http.request", "body": bytes(body), "more_body": False}
                body = None
                return message
            return await receive()

        await self.app(scope, bounded_receive, send)

    @staticmethod
    async def reject(scope, receive, send):
        response = JSONResponse(
            {"detail": f"Request body must be at most {MAX_REQUEST_BYTES} bytes."}, status_code=413,
        )
        await response(scope, receive, send)


def unavailable_detail(setup_command: str):
    return {
        "code": "models_unavailable",
        "message": "Local model artifacts are unavailable. Follow the documented setup for the selected backend, then restart the API.",
        "action": setup_command,
    }


def create_app(evaluator=None, sentiment_evaluator=None, youtube_fetcher=None) -> FastAPI:
    """Inject an offline evaluator in tests; load the real CPU models once at startup."""
    engine = evaluator if evaluator is not None else CommentEvaluator(
        max_tokens=API_MAX_TOKENS, device="cpu", backend=INFERENCE_BACKEND,
    )
    setup_command = getattr(engine, "setup_command", DOWNLOAD_COMMAND)
    sentiment_engine = sentiment_evaluator if sentiment_evaluator is not None else SentimentEvaluator(
        max_tokens=API_MAX_TOKENS, device="cpu",
    )
    sentiment_setup_command = getattr(sentiment_engine, "setup_command", SENTIMENT_DOWNLOAD_COMMAND)
    fetch_comments = youtube_fetcher if youtube_fetcher is not None else fetch_youtube_comments
    youtube_api_key = YOUTUBE_API_KEY
    backend = getattr(engine, "backend", INFERENCE_BACKEND)
    backend = backend if backend in {"public", "shared_bert"} else "unconfigured"
    prediction_lock = Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.models_ready = False
        try:
            engine.load()
            application.state.models_ready = bool(engine.ready)
        except Exception:
            # Startup still succeeds so /health and /predict can explain recovery.
            # Exception details may contain local paths or model-cache information.
            application.state.models_ready = False
        application.state.sentiment_ready = False
        try:
            sentiment_engine.load()
            application.state.sentiment_ready = bool(sentiment_engine.ready)
        except Exception:
            application.state.sentiment_ready = False
        yield

    application = FastAPI(
        title="CommentScope API", version="1.0.0", lifespan=lifespan,
        description="Local positive/neutral/negative sentence sentiment and preserved positive-emotion/toxicity evaluation. The original multi-label mode is not binary sentiment classification. Scores are uncalibrated; comments are not persisted.",
        openapi_tags=[
            {"name": "Evaluation", "description": "Bounded, local single-comment and batch prediction."},
            {"name": "Sentiment", "description": "Independent sentence sentiment model and label-count batch percentages."},
            {"name": "Post comments", "description": "Optional bounded official YouTube comment retrieval."},
            {"name": "Status", "description": "Model readiness and documented input limits."},
        ],
    )
    application.add_middleware(BodySizeLimitMiddleware)
    application.state.models_ready = False
    application.state.sentiment_ready = False

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError):
        # FastAPI normally echoes the rejected value. Omit input/ctx to keep text
        # and accidental secrets out of malformed-request error responses.
        details = []
        for error in exc.errors():
            path = ".".join(str(part) for part in error.get("loc", ()) if part != "body") or "body"
            error_type = error.get("type", "invalid_input")
            message = "Malformed JSON body." if error_type == "json_invalid" else error.get("msg", "Invalid input.")
            if error_type == "extra_forbidden":
                # Unknown key names can themselves contain private input.
                path = "body"
                allowed = {"/predict": "text, texts, threshold, w_pos, w_tox",
                           "/sentiment": "text, texts", "/post-comments/youtube": "url, max_comments"}
                message = "Unknown request field. Allowed fields: " + allowed.get(_request.url.path, "see the API reference") + "."
            details.append({"path": path, "message": message, "code": error_type})
        return JSONResponse({"detail": details}, status_code=422)

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def landing():
        return LANDING_PAGE

    @application.get("/health", tags=["Status"], summary="Check local model readiness")
    def health():
        ready = bool(application.state.models_ready and engine.ready)
        sentiment_ready = bool(application.state.sentiment_ready and sentiment_engine.ready)
        return {
            "status": "ready" if ready else "unavailable", "ready": ready, "backend": backend,
            "sentiment_ready": sentiment_ready, "youtube_configured": bool(youtube_api_key.strip()),
            "limits": {"max_comment_chars": MAX_COMMENT_CHARS, "max_comments": MAX_BATCH_COMMENTS,
                       "max_tokens": getattr(engine, "max_tokens", API_MAX_TOKENS), "max_request_bytes": MAX_REQUEST_BYTES},
            **({} if ready else {"action": setup_command}),
            **({} if sentiment_ready else {"sentiment_action": sentiment_setup_command}),
        }

    @application.post("/predict", tags=["Evaluation"], summary="Evaluate one comment or a bounded batch",
                      response_model=SinglePredictionResponse | BatchPredictionResponse,
                      description="Supply exactly one of text or texts. Each comment is limited to 2,000 characters and 192 model tokens; batches contain at most 20 comments. Inputs are not truncated. Original score/binary/net_goodness keys are retained for single-comment requests.",
                      responses={413: {"description": "Request body exceeds 64 KiB."},
                                 422: {"description": "Invalid input or comment exceeds the token limit."},
                                 503: {"description": "Models unavailable or prediction failed; response contains safe recovery guidance."}})
    def predict(request: PredictionRequest):
        if not application.state.models_ready or not engine.ready:
            raise HTTPException(status_code=503, detail=unavailable_detail(setup_command))
        comments = request.comments()
        try:
            # A single CPU inference at a time bounds peak memory across requests.
            with prediction_lock:
                outputs = engine.predict(comments, threshold=request.threshold, w_pos=request.w_pos, w_tox=request.w_tox)
            if not isinstance(outputs, list) or len(outputs) != len(comments):
                raise PredictionError("Invalid model result count.")
            results = [PredictionResult.model_validate(output).model_dump() for output in outputs]
        except InputTooLongError as exc:
            path = f"texts.{exc.index}" if request.texts is not None else "text"
            raise HTTPException(status_code=422, detail={
                "code": "comment_too_many_tokens", "path": path,
                "message": f"Shorten this comment to at most {exc.max_tokens} model tokens. Input is not truncated.",
            }) from None
        except ModelUnavailableError:
            application.state.models_ready = False
            raise HTTPException(status_code=503, detail=unavailable_detail(setup_command)) from None
        except Exception:
            # Do not return stack traces, submitted text, or exception messages.
            raise HTTPException(status_code=503, detail={
                "code": "prediction_failed", "message": "The local models could not produce a valid result. Retry or restart the API.",
            }) from None

        metadata = {
            "score_type": "uncalibrated_model_scores", "confidence_calibrated": False,
            "description": SCORE_DESCRIPTION, "backend": backend,
            "max_tokens": getattr(engine, "max_tokens", API_MAX_TOKENS), "threshold": request.threshold,
            "w_pos": request.w_pos, "w_tox": request.w_tox,
        }
        if request.texts is not None:
            return {"results": [{"index": index, **result} for index, result in enumerate(results)],
                    "count": len(results), "metadata": metadata}
        return {**results[0], "metadata": metadata}

    @application.post("/sentiment", tags=["Sentiment"], summary="Classify positive, neutral or negative sentence sentiment",
                      response_model=SingleSentimentResponse | BatchSentimentResponse,
                      description="Supply exactly one of text or texts, with at most 20 comments, 2,000 characters and 192 tokens per comment. Batch percentages are counts of predicted labels divided by the number of analyzed comments; they are not model confidence or a platform-wide census.",
                      responses={413: {"description": "Body exceeds 64 KiB."}, 422: {"description": "Invalid input or token limit exceeded."},
                                 503: {"description": "Sentiment model unavailable or prediction failed."}})
    def sentiment(request: SentimentRequest):
        if not application.state.sentiment_ready or not sentiment_engine.ready:
            raise HTTPException(status_code=503, detail=unavailable_detail(sentiment_setup_command))
        comments = request.comments()
        try:
            with prediction_lock:
                outputs = sentiment_engine.predict(comments)
            if not isinstance(outputs, list) or len(outputs) != len(comments):
                raise PredictionError("Invalid sentiment result count.")
            results = [SentimentResult.model_validate(output).model_dump() for output in outputs]
        except InputTooLongError as exc:
            path = f"texts.{exc.index}" if request.texts is not None else "text"
            raise HTTPException(status_code=422, detail={
                "code": "comment_too_many_tokens", "path": path,
                "message": f"Shorten this comment to at most {exc.max_tokens} model tokens. Input is not truncated.",
            }) from None
        except ModelUnavailableError:
            application.state.sentiment_ready = False
            raise HTTPException(status_code=503, detail=unavailable_detail(sentiment_setup_command)) from None
        except Exception:
            raise HTTPException(status_code=503, detail={
                "code": "prediction_failed", "message": "The local sentiment model could not produce a valid result. Retry or restart the API.",
            }) from None
        metadata = {
            "score_type": "uncalibrated_model_scores", "confidence_calibrated": False,
            "max_tokens": getattr(sentiment_engine, "max_tokens", API_MAX_TOKENS),
            "description": "Softmax scores are uncalibrated model outputs. Batch percentages count predicted labels among analyzed comments, not confidence or all platform comments.",
        }
        if request.texts is None:
            return {**results[0], "metadata": metadata}
        count = len(results)
        counts = {label: sum(result["label"] == label for result in results) for label in SENTIMENT_LABELS}
        return {
            "results": [{"index": index, **result} for index, result in enumerate(results)],
            "count": count, "aggregate": {"total": count, "counts": counts,
                                          "percentages": {label: value / count * 100 for label, value in counts.items()}},
            "metadata": metadata,
        }

    @application.post("/post-comments/youtube", tags=["Post comments"], summary="Retrieve up to 100 recent top-level YouTube comments",
                      response_model=YouTubeCommentsResponse,
                      description="Uses a locally configured YOUTUBE_API_KEY and one official API request. Replies, pagination and all-comment census are not included. Returned text is not stored.",
                      responses={422: {"description": "Invalid URL or limit."}, 503: {"description": "YouTube API key is not configured."},
                                 502: {"description": "The upstream service could not provide valid comments."}})
    def youtube_comments(request: YouTubeCommentsRequest):
        try:
            fetched = fetch_comments(request.url, limit=request.max_comments, api_key=youtube_api_key)
            result = YouTubeCommentsResponse.model_validate(fetched)
            if result.received_count > request.max_comments:
                raise PlatformFetchError("Invalid result count.")
            return result
        except InvalidPostURLError:
            raise HTTPException(status_code=422, detail={
                "code": "invalid_post_url", "message": "Enter a public YouTube video URL supported by the documented URL formats.",
            }) from None
        except MissingPlatformAccessError:
            raise HTTPException(status_code=503, detail={
                "code": "youtube_not_configured", "message": "YouTube access is not configured. Add YOUTUBE_API_KEY to the local .env and restart the API, or import comments manually.",
            }) from None
        except PlatformFetchError as exc:
            # This error type exposes only the helper's whitelisted messages.
            status = 503 if exc.reason in {"quotaExceeded", "timeout"} else 502
            raise HTTPException(status_code=status, detail={
                "code": "youtube_fetch_failed", "message": str(exc), "reason": exc.reason,
            }) from None
        except Exception:
            raise HTTPException(status_code=502, detail={
                "code": "youtube_fetch_failed", "message": "YouTube could not provide valid comments. Check video availability, API access or quota, then retry or import comments manually.",
            }) from None

    return application


app = create_app()
