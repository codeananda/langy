# 🎓 Langy - The AI Language Tutor

*Automatically grade, correct, and explain any text in a foreign language.*

Free access here: https://langy-tutor.streamlit.app/

<img src="https://github.com/codeananda/langy/assets/51246969/5f9cfe4e-dbcd-4b85-a071-5b34f8b57730" alt="demo" style="width: 600px; height: auto;">

## 💻 Tech Stack

- Python
- LangChain
- OpenAI
- Streamlit
- Pydantic

## 🔧 Most Recent Changes

- Re-wrote everything using langchain (instead of just the OpenAI API)
- Greatly improved the prompts to deliver more accurate corrections, reduce the number of incorrect corrections, and reduce the number of unhelpful corrections, e.g. "changed X to Y because that is what you do in German" -> "changed X to Y because nouns are always capitalised in German"
- Updated streamlit code to use the streamlit native UI instead of streamlit-chat (much easier to use and more beautiful)
- Moved to its own repo and separated everything into functions/classes

## 🤔 Potential Future Changes

It's *far* from a finished product, there are *loads* of ways it could be improved

- Improve load times. By splitting the prompts up, I've added latency, especially when formatting the corrected text at the end, which hampers UX.
- Add the ability to question the corrections (currently, any text you input will be graded, corrected, and any issues explained). It would be great if the user could ask, "You changed A to B and said the reason is C. Could you have also changed A to Z instead?"
- Add safety checks, e.g. preventing prompt injections and using the moderation API
- Adding an input text limit per request (so that all of the user's text is corrected and graded each time)
- Use GPT-4? This feels like a get-out-of-jail-free card to a certain extent but would obvs make my life easier. I like the idea of pushing GPT-3.5 as hard as possible, though, as it's SO much cheaper and will improve my coding skills more.
