import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import tempfile

client = OpenAI()  # Reads OPENAI_API_KEY from Streamlit Secrets

st.set_page_config(page_title="Kids AI Helper 🌈", page_icon="🌈")
st.title("🌈 Friendly AI Helper for Kids")
st.write("Speak your question and I will answer in a friendly tone!")
# --- Initialize memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    # --- Add user message to memory ---
    st.session_state.chat_history.append({
        "role": "user",
        "content": transcription
    })
    # --- Keep memory short (last 6 messages) ---
    st.session_state.chat_history = st.session_state.chat_history[-10:]

    # --- System Prompt ---
    system_prompt = f"""
You are a very friendly, gentle teacher speaking to a child.

Rules:
- Speak simply and warmly
- Keep answers short and clear
- Be positive and encouraging
- Remember the conversation context
- Respond in: {language}
"""

    # --- Build full message list ---
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(st.session_state.chat_history)

    st.write("💬 Thinking...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )

    answer = response.choices[0].message.content

    # --- Save assistant response to memory ---
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    st.write("💬 **Answer:**")
    st.write(answer)

    # --- Text to Speech ---
    st.write("🔊 Speaking...")

    tts = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=answer
    )

    audio_bytes = tts.read()

    st.audio(audio_bytes, format="audio/mp3")
    st.download_button("⬇️ Download Voice", audio_bytes, "answer.mp3")
    
 
