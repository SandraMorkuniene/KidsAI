import streamlit as st
from openai import OpenAI
import tempfile
from duckduckgo_search import DDGS
import base64

client = OpenAI()

st.title("🎙️ Kids AI Friend")

# Step 1: Record audio
audio_file = st.file_uploader("Ask me something!", type=["wav", "mp3", "m4a"])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Step 2: Speech-to-Text (Whisper)
    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        )
    question = transcript.text
    st.write(f"**You asked:** {question}")

    # Step 3: Web Search
    search_results = DDGS().text(question, max_results=3)
    context = "\n".join([r["body"] for r in search_results])

    # Step 4: LLM Reasoning
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly tutor for children aged 7-12. Explain simply."},
            {"role": "user", "content": f"Question: {question}\n\nInformation:\n{context}"}
        ]
    )
    answer = response.choices[0].message.content
    st.write(f"**Answer:** {answer}")

    # Step 5: Text-to-Speech
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=answer
    )
    audio_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    audio_out.write(speech.read())
    audio_out.flush()

    # Step 6: Play audio
    audio_bytes = open(audio_out.name, "rb").read()
    st.audio(audio_bytes, format="audio/mp3")
