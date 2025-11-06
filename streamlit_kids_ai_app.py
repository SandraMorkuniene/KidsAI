import streamlit as st
from openai import OpenAI
from duckduckgo_search import DDGS
from audiorecorder import audiorecorder
import tempfile
import base64
import os

client = OpenAI()  # Automatically reads OPENAI_API_KEY from Streamlit Secrets

st.set_page_config(page_title="Kids AI Helper 🌈", page_icon="🌈")
st.title("🌈 Friendly AI Helper for Kids")
st.write("Speak your question, and I will help you understand it in a friendly way!")

# --- Language Selector ---
language = st.selectbox("Choose your language:", ["English", "Lithuanian", "Latvian", "Polish"])
lang_code_map = {"English": "en", "Lithuanian": "lt", "Latvian": "lv", "Polish": "pl"}
lang_code = lang_code_map[language]

# --- Record Audio ---
audio = audiorecorder("🎤 Click to record", "🛑 Stop recording")

if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.export(tmp.name, format="wav")
        audio_file_path = tmp.name

    st.write("📝 **Transcribing...**")
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=open(audio_file_path, "rb")
    ).text

    st.write(f"**You said:** {transcript}")

    # --- Decide whether to search ---
    search_check_prompt = f"""
    The child asked: "{transcript}"
    Should we search the web to answer this? Reply ONLY 'yes' or 'no'.
    """
    need_search = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": search_check_prompt}]
    ).choices[0].message.content.strip().lower()

    search_results = ""
    if "yes" in need_search:
        st.write("🔍 Searching...")
        with DDGS() as ddgs:
            results = ddgs.text(transcript, max_results=3)
        for r in results:
            search_results += f"- {r['title']}: {r['body']}\n"

    # --- Generate Kid-Friendly Answer ---
    final_prompt = f"""
    You are a **very friendly, kind teacher** explaining things to a child.
    Speak warmly, clearly, and simply.

    Child question: "{transcript}"
    Web info (may be empty): {search_results}

    Answer in **{language}**, with a calm, soft, reassuring tone.
    """
    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": final_prompt}]
    ).choices[0].message.content

    st.write("💬 **Answer:**")
    st.write(answer)

    # --- Text to Speech (Soft/Warm Voice) ---
    st.write("🔊 Generating voice...")
    tts_audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",  # warm soft tone
        input=answer
    )
    audio_bytes = tts_audio.read()

    st.audio(audio_bytes, format="audio/mp3")

    # Optional Download Button
    st.download_button("⬇️ Download Answer Audio", audio_bytes, "answer.mp3")

    audio_out.flush()

    # Step 6: Play audio
    audio_bytes = open(audio_out.name, "rb").read()
    st.audio(audio_bytes, format="audio/mp3")
