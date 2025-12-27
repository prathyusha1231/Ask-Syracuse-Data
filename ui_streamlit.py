"""
Temporary Streamlit UI for Ask Syracuse Data (test only).
Launch with: streamlit run ui_streamlit.py
"""
from __future__ import annotations
import json
import streamlit as st
import pandas as pd

from main import run_query


st.title("Ask Syracuse Data (Test UI)")
question = st.text_input("Enter a question")

if st.button("Run Query"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        result = run_query(question)

        if result.get("error"):
            st.error(result["error"])
        else:
            st.success("Query executed successfully.")

        st.subheader("Parsed Intent")
        if result.get("intent"):
            st.json(result["intent"])
        else:
            st.write("No intent parsed.")

        st.subheader("Generated SQL")
        st.code(result.get("sql") or "N/A", language="sql")

        st.subheader("Result")
        if isinstance(result.get("result"), pd.DataFrame):
            st.dataframe(result["result"])
        else:
            st.write("No result.")

        st.subheader("Metadata")
        if result.get("metadata"):
            st.json(result["metadata"])
        else:
            st.write("No metadata available.")

        if result.get("limitations"):
            st.info(result["limitations"])
