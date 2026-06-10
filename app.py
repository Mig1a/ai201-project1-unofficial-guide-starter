from __future__ import annotations

import streamlit as st

from scripts.rag import answer_question


st.set_page_config(page_title="The Unofficial Guide", page_icon="?", layout="wide")

st.title("The Unofficial Guide")
st.caption("GMU Computer Science professor review Q&A grounded only in the manually collected review documents.")
st.warning("Answers are based only on the processed Rate My Professors PDF documents in this project.")

question = st.text_input("Ask a question", placeholder="Which professor gives useful feedback?")
model = st.selectbox("Chat model", ["gpt-4o-mini"], index=0)

if st.button("Submit", type="primary"):
    if not question.strip():
        st.error("Enter a question first.")
    else:
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            try:
                result = answer_question(question.strip(), model=model)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.subheader("Answer")
                st.markdown(result["answer"])

                st.subheader("Sources")
                for source in result["sources"]:
                    st.write(f"- {source}")

                st.subheader("Retrieved Chunks")
                for index, chunk in enumerate(result["chunks"], start=1):
                    metadata = chunk["metadata"]
                    label = (
                        f"{index}. {metadata['professor']} - {metadata['source_file']} "
                        f"chunk {metadata['chunk_number']}/{metadata['total_chunks']}"
                    )
                    with st.expander(label):
                        st.json(metadata)
                        st.write(chunk["text"])
