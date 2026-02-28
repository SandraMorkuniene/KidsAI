import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import tempfile

client = OpenAI()  # Reads OPENAI_API_KEY from Streamlit Secrets

st.set_page_config(page_title="Kids AI Helper 🌈", page_icon="🌈")
st.title("🌈 Friendly AI Helper for Kids")
st.write("Speak your question and I will answer in a friendly tone!")

# --- Language selection ---
language = st.selectbox("Language:", ["Lithuanian", "English"])

# --- Microphone Recorder ---
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

    # --- STEP 1: Decide Knowledge Mode ---
    st.write("🧠 Thinking...")

    knowledge_check = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""
You are deciding how to answer a child's question.

Question: "{transcription}"

If this requires recent facts (like current news, weather, or events after 2023),
reply ONLY with: RECENT

Otherwise reply ONLY with: GENERAL
"""
            }
        ],
        temperature=0
    ).choices[0].message.content.strip()

    # --- STEP 2: Generate Answer ---
    answer_prompt = f"""
You are a very friendly, gentle teacher speaking to a child.

Rules:
- Speak simply and warmly
- Keep answers short and clear
- Be positive and encouraging
- If unsure about very recent events, say kindly that you may not know the newest updates
- Respond in: {language}

Child asked:
"{transcription}"

Knowledge type: {knowledge_check}
"""

    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": answer_prompt}],
        temperature=0.7
    ).choices[0].message.content

    st.write("💬 **Answer:**")
    st.write(answer)

    # --- Text to Speech ---
    st.write("🔊 Speaking...")

    tts = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",  # warm & soft
        input=answer
    )

    audio_bytes = tts.read()

    st.audio(audio_bytes, format="audio/mp3")
    st.download_button("⬇️ Download Voice", audio_bytes, "answer.mp3")
