import openai
import streamlit as st
from redlines import Redlines

from utils import (
    classify_text_level,
    correct_text,
    parse_corrections,
)

openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.organization = st.secrets["OPENAI_ORG_ID"]

# Setting page title and header
title = "Langy - The AI Language Tutor"
st.set_page_config(page_title=title, page_icon=":mortar_board:")
st.title(":mortar_board: " + title)

# Intro
intro = """*Input text in a foreign language, get corrections + explanations out.*
"""
st.markdown(intro)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Let user clear the current conversation
clear_button = st.button("Clear Conversation", key="clear")
if clear_button:
    st.session_state["messages"] = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Accept user input
if prompt := st.chat_input("Enter some text to get corrections"):
    main(prompt)
