import openai
import streamlit as st
from redlines import Redlines

from helpers import (
    classify_text_level,
    correct_text,
    parse_corrections,
)

openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.organization = st.secrets["OPENAI_ORG_ID"]

MODEL_TOKEN_LIMIT = 4000

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
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        text_class = classify_text_level(prompt, message_placeholder)
        text_correct = correct_text(
            prompt, message_placeholder, message_contents=text_class + "\n\n"
        )
        text_correct = parse_corrections(text_correct)
        comparison = Redlines(prompt, text_correct.corrected_text)
        comparison = comparison.output_markdown

        final_response = f"{text_class}\n\n"
        final_response += "## Corrected Text\n\n"
        final_response += f"{comparison}\n\n"
        final_response += "## Reasons\n\n"
        for reason in text_correct.reasons:
            final_response += f"1. {reason}\n"

        message_placeholder.empty()
        message_placeholder.markdown(final_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
