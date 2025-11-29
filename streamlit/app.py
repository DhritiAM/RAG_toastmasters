import streamlit as st
import json
import os
import sys
from feedback_db import init_db, save_feedback

# Add parent directory to path to import RAG pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rag_pipeline.main import RAG

# Initialize database
init_db()

# Initialize RAG pipeline (cache in session_state to avoid reloading on every rerun)
if "rag" not in st.session_state:
    with st.spinner("Loading RAG pipeline..."):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        st.session_state.rag = RAG(config_path=config_path, verbose=False)

st.title("TM Knowledge Assistant ✨")
st.write("Ask anything or try a suggested question!")

# Load suggestions from local JSON file
json_path = os.path.join(os.path.dirname(__file__), "suggested_questions.json")
with open(json_path, "r", encoding="utf-8") as f:
    suggestions = json.load(f)["suggestions"]

# Display suggestions
st.subheader("Suggested Questions")
cols = st.columns(2)

for i, s in enumerate(suggestions):
    if cols[i % 2].button(s):
        st.session_state.query = s

# Text input for question
query = st.text_input("Your Question:", value=st.session_state.get("query", ""))

if st.button("Submit"):
    if query:
        st.session_state.query = query  # Store the query in session_state
        with st.spinner("Thinking..."):
            result = st.session_state.rag.query(query)
            st.session_state.answer = result["response"]
    else:
        st.warning("Please enter a question.")

# Display answer
if "answer" in st.session_state:
    st.subheader("Answer")
    st.write(st.session_state.answer)

    st.subheader("Was this helpful?")
    rating = st.slider("Rate the answer (1 = bad, 5 = great)", 1, 5, 3)
    comments = st.text_area("Additional comments:")

    if st.button("Submit Feedback"):
        # Use the query from session_state to ensure consistency
        feedback_query = st.session_state.get("query", query)
        save_feedback(feedback_query, st.session_state.answer, rating, comments)
        st.success("Your feedback has been saved! 🌿")
