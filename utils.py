from textwrap import dedent
from typing import Any

import streamlit as st
from langchain import PromptTemplate, ConversationChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import LLMResult
from pydantic import BaseModel, Field
from redlines import Redlines

MODEL_TOKEN_LIMIT = 4000
MODEL_NAME = 'gpt-4'

class StreamingStreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(
        self,
        message_placeholder: st.delta_generator.DeltaGenerator,
        message_contents: str = "",
    ):
        """Initialize the callback handler.

        Parameters
        ----------
        message_placeholder: st.delta_generator.DeltaGenerator
            The placeholder where the messages will be streamed to. Typically an st.empty() object.
        """
        self.message_placeholder = message_placeholder
        self.message_contents = message_contents

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.message_contents += token
        self.message_placeholder.markdown(self.message_contents + "▌")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.message_placeholder.markdown(self.message_contents)


def classify_text_level(prompt, message_placeholder) -> str:
    """Classify the prompt based on the Common European Framework of Reference. Prompt
    is assumed to be text in a foreign language that the user wants help with."""
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0,
        streaming=True,
        callbacks=[StreamingStreamlitCallbackHandler(message_placeholder)],
    )

    shorter_template = """Classify the text based on the Common European Framework of Reference
    for Languages (CEFR), provide maxiumum 50 words for your answer.

    Text: {text}

    Format the output as markdown like this:

    ## CEFR Level: <level>
    
    <reason>
     """
    prompt_template_reason_level = ChatPromptTemplate(
        messages=[HumanMessagePromptTemplate.from_template(shorter_template)],
        input_variables=["text"],
    )

    chain_reason_level = LLMChain(
        llm=llm, prompt=prompt_template_reason_level, output_key="reason_level"
    )

    response = chain_reason_level({"text": prompt})
    # Add cefr_text explanation to bottom
    cefr_text = (
        "\n\nSee [Common European Framework of Reference for Languages]"
        "(https://en.wikipedia.org/wiki/Common_European_Framework_of_Reference_for_Languages)"
        " for more information on language levels."
    )
    for letter in cefr_text:
        response["reason_level"] += letter
        message_placeholder.markdown(response["reason_level"] + "▌")
    message_placeholder.markdown(response["reason_level"])
    return response["reason_level"]


def correct_text(prompt, message_placeholder, message_contents="") -> str:
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0,
        streaming=True,
        callbacks=[
            StreamingStreamlitCallbackHandler(
                message_placeholder, message_contents=message_contents
            )
        ],
    )
    correction_template = """The following is a friendly conversation between a human and an AI. The
    AI is helping the human improve their foreign language writing skills. The human provides texts
    written in a foreign language and the AI corrects the spelling and grammar of the texts
    and provides detailed reasons for each correction.

    The AI keeps in in mind spelling, grammar, naturalness (how much it sounds like a native
    speaker), correct capitalisation, correct placement of commas or other punctuation and
    anything else necessary for correct writing.

    The AI only provides corrections for words/phrases that have changed. If the original
    text is the same as the corrected text, then the AI does not provide a correction.

    The AI knows that each sentence may contain multiple errors and provides corrections for
    all errors in the sentence. It also knows that some sentences will not contain any errors
    and does not provide corrections for those sentences.

    The AI does not give answers like "changed X to Y because this is how it is done in German".
    Instead, it explains the reason for the change, e.g. "changed X to Y because Z".
    
    The AI only gives one explanation for each change. It does not repeat the same explanation
    multiple times.
    
    The AI counts all changes such as "changed X to Y" as one change. If there are multiple reasons
    for the change, they are listed in the same bullet point. For example, "changed X to Y because
    Z and W".    

    If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {history}
    Human: {input}
    AI: Let's think step by step"""
    correction_prompt = PromptTemplate(
        input_variables=["history", "input"], template=correction_template
    )

    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=MODEL_TOKEN_LIMIT)

    input_1 = "Hallo, ich heisse Adam. Ich habe 25 Jahre alt."
    output_1 = dedent(
        """
    Let's think step by step
    ## Corrected Text

    Ich heiße Adam. Ich bin 25 Jahre alt.

    ## Reasons
    1. Corrected spelling of 'heisse' to 'heiße' because 'ss' can be combined to form 'ß' in German.
    2. Corrected 'alt' to 'bin' because 'bin' is the correct verb to use when stating one's age in German."""
    )
    memory.save_context({"input": input_1}, {"output": output_1})

    input_2 = "Ich bin 25 Jahre alt"
    output_2 = dedent(
        """
    Let's think step by step
    ## Corrected Text

    Ich bin 25 Jahre alt.

    ## Reasons
    1. Added full stop to the end of the sentence because it is a complete sentence."""
    )
    memory.save_context({"input": input_2}, {"output": output_2})

    input_3 = "Ich habe eine Katze. Sie ist schwarz und klein."
    output_3 = dedent(
        """
    Let's think step by step
    ## Corrected Text

    Ich habe eine Katze. Sie ist schwarz und klein.

    ## Reasons
    1. No corrections needed. The text is grammatically correct and natural."""
    )

    memory.save_context({"input": input_3}, {"output": output_3})

    input_4 = "Ich wohne auf England fuer 15 Jahren."
    output_4 = dedent(
        """
    Let's think step by step
    ## Corrected Text

    Ich wohne in England seit 15 Jahren.

    ## Reasons
    1. Corrected 'auf' to 'in' because 'in' is the correct preposition to use when talking about living in a country.
    2. Corrected 'fuer' to 'seit' because 'seit' is the correct preposition to use when talking about the duration of time.
    """
    )
    memory.save_context({"input": input_4}, {"output": output_4})

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False,
        prompt=correction_prompt,
    )

    response = conversation.predict(input=prompt)
    return response


def parse_corrections(correction_and_reasons):
    """Extract the corrections/reasons from input and store in Pydantic object."""
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0,
    )

    template = """Extract the corrections and reasons for them from the text.

    Text: ####{text}####

    {format_instructions}
     """

    class Output(BaseModel):
        corrected_text: str = Field(description="The corrected text (no heading)")
        reasons: list[str] = Field(description="The list of reasons.")

    parser = PydanticOutputParser(pydantic_object=Output)

    prompt_template = ChatPromptTemplate(
        messages=[HumanMessagePromptTemplate.from_template(template)],
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = LLMChain(llm=llm, prompt=prompt_template, output_key="output")

    output = chain({"text": correction_and_reasons})
    results = parser.parse(output["output"])
    return results

def main(prompt):
    """Classify, correct and explain the text."""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Classify
        text_class = classify_text_level(prompt, message_placeholder)
        # Correct + parse
        text_correct = correct_text(
            prompt, message_placeholder, message_contents=text_class + "\n\n"
        )
        text_correct = parse_corrections(text_correct)
        # Compare with input and create nice redline formatting of changes
        comparison = Redlines(prompt, text_correct.corrected_text)
        comparison = comparison.output_markdown

        # Combine all results into one string and display
        final_response = f"{text_class}\n\n"
        final_response += "## Corrected Text\n\n"
        final_response += f"{comparison}\n\n"
        final_response += "## Reasons\n\n"
        for reason in text_correct.reasons:
            final_response += f"1. {reason}\n"

        message_placeholder.markdown(final_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": final_response})