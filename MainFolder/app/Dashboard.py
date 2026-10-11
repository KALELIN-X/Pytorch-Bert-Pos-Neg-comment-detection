"""Local sentence and post-comment analysis with independent evaluation tasks."""

from html import escape
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.comment_import import aggregate_sentiment, parse_comment_csv, parse_pasted_comments, validate_comments
from src.schemas import BatchPredictionResponse, SinglePredictionResponse, BatchSentimentResponse, SingleSentimentResponse, YouTubeCommentsResponse

load_dotenv(ROOT / '.env', override=False)
load_dotenv(ROOT / 'MainFolder' / '.env', override=False)
API_URL = os.getenv('API_URL', 'http://127.0.0.1:8000/predict')
parts = urlsplit(API_URL)


def endpoint(path):
    return urlunsplit((parts.scheme, parts.netloc, path, '', ''))


SAMPLES = {
    'Appreciation': 'Thank you for explaining this so clearly. I really appreciate your help!',
    'Hostility': "You are an idiot. Shut up and stop wasting everyone's time.",
    'Factual': "The meeting starts at three o'clock.",
    'Ambiguous': 'Great, another meeting that could have been an email.',
}
LABEL_NAMES = {
    'positive_signals': 'Positive emotion signals', 'toxicity_signals': 'Toxicity signals',
    'mixed_signals': 'Mixed signals', 'no_detected_signals': 'No signals above threshold',
}

st.set_page_config(page_title='CivicBalance | Comment evaluator', page_icon='💬', layout='wide')
st.markdown('''
<style>
.block-container { max-width: 1160px; padding-top: 2rem; padding-bottom: 3rem; }
h1 { font-size: 2.3rem !important; letter-spacing: -.05rem; }
h2 { font-size: 1.4rem !important; }
[data-testid="stSidebar"] { border-right: 1px solid #dde5e4; }
.eyebrow { color: #14756d; font-size: .75rem; font-weight: 700; letter-spacing: .12rem; }
.result-heading { font-size: 1.4rem; font-weight: 650; margin: .2rem 0 .6rem; }
.score-row { display: grid; grid-template-columns: 110px minmax(60px, 1fr) 46px; gap: 12px; align-items: center; margin: 14px 0; font-size: .88rem; }
.score-track { height: 7px; border-radius: 5px; background: #e8eeed; overflow: hidden; }
.score-fill { height: 100%; border-radius: 5px; }
.score-value { color: #526365; text-align: right; font-variant-numeric: tabular-nums; }
@media(max-width: 700px) { .block-container { padding-top: 1.3rem; } h1 { font-size: 1.9rem !important; } }
</style>
''', unsafe_allow_html=True)


def api_error(response, comment_offset=0):
    try:
        detail = response.json().get('detail', 'The API could not evaluate these comments.')
    except (ValueError, AttributeError):
        return f'The API returned an unexpected response (HTTP {response.status_code}).'
    if isinstance(detail, dict):
        path = str(detail.get('path', ''))
        prefix = ''
        if path.startswith('texts.') and path[6:].isdigit():
            prefix = f'Comment {comment_offset + int(path[6:]) + 1}: '
        return prefix + str(detail.get('message', 'Evaluation failed.')) + (' Setup: ' + str(detail['action']) if detail.get('action') else '')
    if isinstance(detail, list) and all(isinstance(item, dict) for item in detail):
        return ' '.join(f"{item.get('path', 'Input')}: {item.get('message', 'Invalid input.')}" for item in detail)
    return 'The API could not evaluate this input.'


def post_json(url, payload, schema=None, comment_offset=0):
    response = requests.post(url, json=payload, timeout=(5, 120))
    if not response.ok:
        raise RuntimeError(api_error(response, comment_offset))
    data = response.json()
    return schema.model_validate(data).model_dump() if schema else data


def score_rows(scores, color):
    rows = []
    for label, value in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        width = min(100, max(0, value * 100))
        rows.append(f'<div class="score-row"><span>{escape(label.replace("_", " ").title())}</span>'
                    f'<div class="score-track"><div class="score-fill" style="width:{width:.2f}%;background:{color}"></div></div>'
                    f'<span class="score-value">{value:.3f}</span></div>')
    st.markdown(''.join(rows), unsafe_allow_html=True)


health = {}
with st.sidebar:
    st.markdown('### Evaluation task')
    task = st.radio('What to evaluate', ['Sentiment', 'Emotions & toxicity'])
    sentiment = task == 'Sentiment'
    if sentiment:
        st.caption('Positive, neutral or negative sentiment. The label with the highest model score is selected.')
        st.caption('DistilBERT sentiment model · local CPU inference')
    else:
        st.caption('The original project workflow: six positive emotions and six toxicity labels. Several labels can be detected at once.')
        threshold = st.slider('Detection threshold', 0.0, 1.0, 0.5, 0.01)
        with st.expander('Heuristic score weights'):
            w_pos = st.number_input('Positive emotion weight', min_value=0.0, max_value=3.0, value=1.0, step=0.1)
            w_tox = st.number_input('Toxicity weight', min_value=0.0, max_value=3.0, value=1.2, step=0.1)
            st.caption('Net goodness = weighted mean positive score − weighted mean toxicity score. This is a tunable heuristic.')
    st.divider()
    try:
        status_response = requests.get(endpoint('/health'), timeout=2)
        status_response.raise_for_status()
        health = status_response.json()
        if not isinstance(health, dict):
            raise ValueError('Invalid health response.')
        ready = health.get('sentiment_ready') if sentiment else health.get('ready')
        if ready:
            st.success('Models ready')
        else:
            st.warning('Models need setup')
            action = health.get('sentiment_action', 'python -m src.download_sentiment') if sentiment else health.get('action', 'python -m src.download_models')
            st.code(action, language='bash')
    except (requests.RequestException, ValueError, AttributeError):
        st.caption('API offline. Start the FastAPI server to evaluate comments.')
    st.caption('Scores are uncalibrated model outputs, not confidence percentages. Comments are not stored by the application.')
    st.link_button('Open API documentation', endpoint('/docs'), width='stretch')

st.markdown('<div class="eyebrow">CIVICBALANCE / COMMENT EVALUATOR</div>', unsafe_allow_html=True)
st.title('Comment evaluator')
st.write('Evaluate a sentence or explore the comments on a post.')
mode = st.radio('Evaluation mode', ['One comment', 'Post comments'], horizontal=True)
limit = 100 if sentiment else 20
source = 'Paste comments'
if mode == 'Post comments':
    source = st.radio('Comment source', ['Paste comments', 'CSV upload', 'YouTube URL'], horizontal=True)
    st.caption(f'Analyze up to {limit} comments. Percentages summarize only the supplied or retrieved comments.')

if source == 'Paste comments':
    if mode == 'One comment':
        buttons = st.columns(len(SAMPLES))
        for column, (name, sample) in zip(buttons, SAMPLES.items()):
            with column:
                if st.button(name, width='stretch'):
                    st.session_state['comment_input'] = sample
        st.caption('Ambiguous language and sarcasm may produce unexpected labels.')
    if 'comment_input' not in st.session_state:
        st.session_state['comment_input'] = SAMPLES['Appreciation']

with st.form('evaluate'):
    text, uploaded, post_url = '', None, ''
    if mode == 'One comment' or source == 'Paste comments':
        text = st.text_area('Comment' if mode == 'One comment' else 'Comments — one per line', key='comment_input', height=130,
                            help=f'Up to 2,000 characters and 192 model tokens per comment; a trained checkpoint may impose a smaller token limit. Post mode supports {limit} comments; blank separator lines are skipped.')
    elif source == 'CSV upload':
        uploaded = st.file_uploader('Comments CSV', type=['csv'], help='UTF-8 CSV, at most 256 KiB. Include exactly one column named text, comment, or comment_text. Comments can contain quoted newlines.')
        st.caption('Use an authorized comment export. Files are processed in memory.')
    else:
        post_url = st.text_input('YouTube video URL', placeholder='https://www.youtube.com/watch?v=...')
        st.caption('Uses the official YouTube API. Fetches recent top-level comments; replies are excluded.')
        if not health.get('youtube_configured'):
            st.warning('YouTube needs a local YOUTUBE_API_KEY. Pasted and CSV comments work without one.')
    submitted = st.form_submit_button('Evaluate comment' if mode == 'One comment' else ('Fetch & evaluate' if source == 'YouTube URL' else 'Evaluate comments'), type='primary', width='stretch')

if submitted:
    st.session_state.pop('evaluation', None)
    comments = []
    try:
        scope_note = 'Comments supplied by you; this is not a count of every comment on the post.'
        if mode == 'One comment':
            comments = validate_comments([text], 1)
        elif source == 'Paste comments':
            comments = parse_pasted_comments(text, limit)
        elif source == 'CSV upload':
            if uploaded is None:
                raise ValueError('Choose a comments CSV file before evaluating.')
            comments = parse_comment_csv(uploaded.getvalue(), limit)
        else:
            with st.spinner('Fetching permitted YouTube comments…'):
                fetched = post_json(endpoint('/post-comments/youtube'), {'url': post_url, 'max_comments': limit}, YouTubeCommentsResponse)
            if not fetched['comments']:
                raise ValueError('No nonblank comments within the input limits were retrieved.')
            comments = validate_comments(fetched['comments'], limit)
            scope_note = (f"Sample of {fetched['received_count']} recent top-level YouTube comments; replies excluded. "
                          f"{fetched['skipped_blank']} blank and {fetched['skipped_oversized']} oversized comments excluded. "
                          + ('More comments are available beyond this sample.' if fetched['has_more'] else ''))
        results, metadata = [], None
        with st.spinner('Evaluating locally…'):
            for start in range(0, len(comments), 20):
                chunk = comments[start:start + 20]
                is_single = mode == 'One comment'
                payload = {'text': chunk[0]} if is_single else {'texts': chunk}
                if sentiment:
                    schema = SingleSentimentResponse if is_single else BatchSentimentResponse
                    data = post_json(endpoint('/sentiment'), payload, schema, comment_offset=start)
                else:
                    payload.update(threshold=threshold, w_pos=w_pos, w_tox=w_tox)
                    schema = SinglePredictionResponse if is_single else BatchPredictionResponse
                    data = post_json(API_URL, payload, schema, comment_offset=start)
                chunk_results = [data] if is_single else data['results']
                if len(chunk_results) != len(chunk) or (not is_single and (
                    data['count'] != len(chunk) or [result['index'] for result in chunk_results] != list(range(len(chunk)))
                )):
                    raise ValueError('The API returned an incomplete result.')
                results.extend(chunk_results)
                metadata = data['metadata']
        st.session_state['evaluation'] = {
            'task': task, 'comments': comments, 'results': results, 'metadata': metadata,
            'aggregate': aggregate_sentiment(results) if sentiment else None,
            'scope_note': scope_note, 'mode': mode, 'source': source,
        }
    except requests.Timeout:
        st.error('The local models took too long to respond. Try a smaller set or check the API terminal.')
    except requests.RequestException:
        st.error('Cannot reach the API. Start the FastAPI server and try again.')
    except RuntimeError as error:
        st.error(str(error))
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        # Pydantic errors may contain rejected response values; never display them.
        message = str(error) if type(error) is ValueError else 'The API returned an incomplete result. Check the dashboard and API versions.'
        st.error(message)
    if comments and 'evaluation' not in st.session_state and mode == 'Post comments':
        st.caption('The full set could not be evaluated, so no percentage summary is shown. Review the numbered comments and correct the input before retrying.')
        with st.expander('Comments to review'):
            st.dataframe([{'Comment number': index + 1, 'Text': comment} for index, comment in enumerate(comments)], hide_index=True, width='stretch')

saved = st.session_state.get('evaluation')
if saved and saved['task'] == task and saved['mode'] == mode and saved.get('source', 'Paste comments') == source:
    st.divider()
    st.subheader('Evaluation results')
    metadata = saved['metadata']
    if sentiment:
        if saved['mode'] == 'Post comments':
            summary = saved['aggregate']
            st.caption(f"{summary['total']} comments analyzed. {saved['scope_note']}")
            for column, label in zip(st.columns(3), ['positive', 'neutral', 'negative']):
                with column:
                    st.metric(label.title(), f"{summary['percentages'][label]:.1f}%")
                    st.caption(f"{summary['counts'][label]} of {summary['total']} comments")
            st.caption('Percentages are shares of predicted labels, not confidence scores or verified ground truth. Rounding can make the displayed total differ slightly from 100%.')
            rows = [{'Comment': comment, 'Predicted sentiment': result['label'].title(),
                     **{label.title() + ' score': result['scores'][label] for label in ['positive', 'neutral', 'negative']}}
                    for comment, result in zip(saved['comments'], saved['results'])]
            st.dataframe(rows, hide_index=True, width='stretch',
                         column_config={label.title() + ' score': st.column_config.NumberColumn(format='%.3f') for label in ['positive', 'neutral', 'negative']})
        else:
            result = saved['results'][0]
            with st.container(border=True):
                st.markdown(f'<div class="result-heading">{escape(result["label"].title())} sentiment</div>', unsafe_allow_html=True)
                st.write(result['explanation'])
                st.caption('Evaluated text: ' + saved['comments'][0])
                score_rows(result['scores'], '#178b7f')
        st.caption('Softmax model scores are uncalibrated; they are not confidence percentages. Sentiment analysis is separate from the original emotion/toxicity workflow.')
        with st.expander('API results'):
            st.json({'results': saved['results'], 'aggregate': saved['aggregate'], 'metadata': metadata})
    else:
        st.caption(f"Results at threshold {metadata['threshold']:.2f}. Model scores are uncalibrated; they are not confidence percentages.")
        for index, (comment, result) in enumerate(zip(saved['comments'], saved['results'])):
            with st.container(border=True):
                if len(saved['results']) > 1:
                    st.caption(f'COMMENT {index + 1}')
                st.markdown(f'<div class="result-heading">{escape(LABEL_NAMES.get(result["label"], result["label"]))}</div>', unsafe_allow_html=True)
                st.write(result['explanation'])
                st.caption('Evaluated text: ' + comment)
                positive, toxic = st.columns(2, gap='large')
                with positive:
                    st.markdown('**Positive emotions**')
                    score_rows(result['positivity'], '#178b7f')
                with toxic:
                    st.markdown('**Toxicity**')
                    score_rows(result['toxicity'], '#c06355')
                st.caption(f"Heuristic net goodness: {result['net_goodness']:.3f} · positive weight {metadata['w_pos']:.1f} · toxicity weight {metadata['w_tox']:.1f}")
                with st.expander('API response'):
                    st.json(result)
else:
    st.info('Enter a comment or provide post comments, then evaluate to see the results.')
