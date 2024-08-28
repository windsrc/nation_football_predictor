import streamlit as st
import pandas as pd

    
# ----- Page configs (tab title, favicon) -----
st.set_page_config(
    page_title="National Football Predictor - Statistics",
    page_icon="⚽️",
)


# ----- Left menu -----
with st.sidebar:
    st.header("National Football Team Prediction Tool")
    st.write("###")
    st.write("**Author:** [Christopher Windsor](https://github.com/windsrc)")


# ----- Top title -----
st.write(f"""<div style="text-align: center;"><h1 style="text-align: center;">How does the tool work?</h1></div>""", unsafe_allow_html=True)