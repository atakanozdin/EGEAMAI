import streamlit as st
from functions_ui import *

# STREAMLIT PAGE
tabs = ["Data Analyst", "Researcher", "ML Engineer", "AM Engineer", "Test Center", "EGE AMAI LIFT-UP TEAM Hakkında"]
page = st.sidebar.radio("Sayfalar", tabs)


# Data Analys Ekranı
if page == "Data Analyst":
    DataAnalystRun()

# Researcher Ekranı
if page == "Researcher":
    st.success("EGE AMAI LIFT-UP Researcher Part")


# ML Engineer Ekranı
if page == "ML Engineer":
    st.success("EGE AMAI LIFT-UP ML Engineer Part")


# AM Engineer Ekranı
if page == "AM Engineer":
    AMEngineerRun()


if page=="Test Center":
    st.success("LLM RAG Model Test Center")


if page == "EGE AMAI LIFT-UP TEAM Hakkında":
    st.success("EGE AMAI LIFT-UP TEAM")
    st.write("................................")
