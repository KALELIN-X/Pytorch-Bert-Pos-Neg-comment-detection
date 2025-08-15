import streamlit as st
import requests
import os

API_URL = os.getenv('API_URL', 'http://localhost:8000/predict')

st.set_page_config(page_title="CivicBalance", layout="wide")
st.title("CivicBalance — Promote Good, Reduce Harm")

col1, col2 = st.columns(2)
with col1:
    text = st.text_area("Paste text", height=200, value="I really appreciate your help!")
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
    w_pos = st.number_input("w_pos", value=1.0, step=0.1)
    w_tox = st.number_input("w_tox", value=1.2, step=0.1)
    if st.button("Score"):
        r = requests.post(API_URL, json={"text": text, "threshold": threshold, "w_pos": w_pos, "w_tox": w_tox})
        if r.ok:
            st.session_state['res'] = r.json()
        else:
            st.error(r.text)

res = st.session_state.get('res')
if res:
    with col2:
        st.subheader("Net Goodness Score")
        st.metric("NGS", f"{res['net_goodness']:.3f}")
        st.subheader("Positivity")
        st.json(res['positivity'])
        st.subheader("Toxicity")
        st.json(res['toxicity'])