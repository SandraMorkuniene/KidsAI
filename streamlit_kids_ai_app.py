import streamlit as st
from openai import OpenAI
from duckduckgo_search import DDGS
from streamlit_mic_recorder import mic_recorder
import tempfile
import os

client = OpenAI()  # Reads OPENAI_API_KEY from Streamlit Secrets

st.set_page_config(page_title="Kids AI Helper 🌈", page_icon="🌈")
st.title("🌈 Friendly AI Helper for Kids")
st.write("Speak your question and I will answer in a friendly tone!")

# --- Language selection ---
language = st.selectbox("Language:", ["English", "Lithuanian", "Latvian", "Polish"])
lang_map = {"English": "en", "Lithuanian": "lt", "Latvian": "lv", "Polish": "pl"}
lang = lang_map[language]

# --- Microphone Recorder (No ffmpeg needed ✅) ---
audio = mic_recorder(
    start_prompt="🎤 Click to start recording",
    stop_prompt="🛑 Stop recording",
    key="recorder"
)

if audio:
    st.audio(audio["bytes"])

    # Save recorded WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio["bytes"])
        wav_file = tmp.name

    st.write("📝 Transcribing...")

    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=open(wav_file, "rb")
    ).text

    st.write(f"**You said:** {transcription}")

    # Decide whether search is needed
    check = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": f"""
        The child asked: "{transcription}"
        Should we look up information online?
        Reply only 'yes' or 'no'.
        """}]
    ).choices[0].message.content.strip().lower()

    info = ""
    if "yes" in check:
        st.write("🔍 Searching...")
        with DDGS() as ddgs:
            results = ddgs.text(transcription, max_results=3)
            for r in results:
                info += f"{r['title']}: {r['body']}\n"

    # Generate friendly answer
    st.write("💬 Thinking...")

    answer_prompt = f"""
    You are a very friendly, gentle teacher speaking to a child.
    Speak simply, warmly, and kindly.
    
    Language: {language}
    Child asked: "{transcription}"
    Extra info (optional): {info}
    """

    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": answer_prompt}]
    ).choices[0].message.content

    st.write("💬 **Answer:**")
    st.write(answer)

    # Text → Speech (Soft/Warm Voice)
    st.write("🔊 Speaking...")

    tts = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",  # soft & warm
        input=answer
    )

    audio_bytes = tts.read()
    st.audio(audio_bytes, format="audio/mp3")

    st.download_button("⬇️ Download Voice", audio_bytes, "answer.mp3")
