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

# --- Reset conversation button ---
if st.button("🔄 Start New Conversation"):
    st.session_state.chat_history = []
    st.success("Conversation reset!")

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

    #st.write("📝 Transcribing...")

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
    # --- Keep memory short (last 10 messages) ---
    st.session_state.chat_history = st.session_state.chat_history[-10:]

    st.write("💬 Thinking...")

    # --- Generate Answer with Web Search + Memory ---
    response = client.responses.create(
        model="gpt-4.1-mini",  # Supports web_search
        tools=[{"type": "web_search"}],
        input=[
            {
                "role": "system",
                "content": f"""
You are a very friendly, gentle teacher speaking to a child.

Rules:
- Speak simply and warmly
- Keep answers clear and not too long
- Be encouraging and positive
- If using web search results, explain them simply
- Respond in: {language}
"""
            },
            *st.session_state.chat_history
        ]
    )

    answer = ""
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    answer += content.text

    # --- Save assistant reply to memory ---
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

