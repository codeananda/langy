import openai
import streamlit as st

from utils import main

openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.organization = st.secrets["OPENAI_ORG_ID"]

# Setting page title and header
title = "Langy - The AI Language Tutor"
st.set_page_config(page_title=title, page_icon=":mortar_board:")
st.title(":mortar_board: " + title)

# Intro
intro = "*Input text in a foreign language, get corrections and explanations out.*"
st.markdown(intro)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

columns = st.columns([1, 1, 1, 1])
with columns[0]:
    # Give the user an example
    example_button = st.button('Give Me An Example', key="example")

with columns[1]:
    # Let user clear the current conversation
    clear_button = st.button("Clear Conversation", key="clear")
    if clear_button:
        st.session_state["messages"] = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Run simple example if button is clicked
if example_button:
    main('Hallo, ich heisse Langy. Ich habe 25 Jahren alt.')

# Accept user input
if prompt := st.chat_input("Enter some text to get corrections"):
    main(prompt)
