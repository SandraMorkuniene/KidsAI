import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
import tempfile

st.set_page_config(page_title="Kids AI 🌈", page_icon="🌈")

client = OpenAI()  # Reads OPENAI_API_KEY from Streamlit Secrets

# --- Language selection ---
language = st.selectbox("Language:", ["Lithuanian", "English"])


translations = {
    "English": {
        "title": "🌈 Kids AI",
        "subtitle": "Shoot your question!",
        "language_label": "Language:",
        "new_chat": "🔄 Start New Conversation",
        "new_chat_success": "Conversation reset!",
        "download_chat": "⬇️ Download Conversation",
        "thinking": "💬 Let me think!",
        "speaking": "🔊 Speaking...",
        "record_start": "🎤 Click to start recording",
        "record_stop": "🛑 Stop recording"
    },
    "Lithuanian": {
        "title": "🌈 Vaikų AI",
        "subtitle": "Užduok klausimą!",
        "language_label": "Kalba:",
        "new_chat": "🔄 Naujas pokalbis",
        "new_chat_success": "Pradėkim iš naujo!",
        "download_chat": "⬇️ Atsisiųsti pokalbį",
        "thinking": "💬 Galvoju...",
        "speaking": "🔊 Atsakau...",
        "record_start": "🎤 Spausk ir pradėk kalbėti",
        "record_stop": "🛑 Sustabdyti įrašą"
    }
}

ui = translations[language]


st.title(ui["title"])
st.write(ui["subtitle"])

# --- Initialize memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Reset conversation button ---
if st.button(ui["new_chat"]):
    st.session_state.chat_history = []
    st.rerun()
    #st.success("Conversation reset!")    



# Map UI language to ISO code
lang_map = {
    "English": "en",
    "Lithuanian": "lt"
}

selected_lang_code = lang_map[language]

for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧒"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message["content"])
            

# --- Microphone Recorder ---
audio = mic_recorder(
    start_prompt=ui["record_start"],
    stop_prompt=ui["record_stop"],
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
    file=open(wav_file, "rb"),
        language=selected_lang_code  # force language
    ).text

    #st.write(f"**You said:** {transcription}")

    # --- Add user message to memory ---
    st.session_state.chat_history.append({
        "role": "user",
        "content": transcription
    })
    # --- Keep memory short (last 10 messages) ---
    st.session_state.chat_history = st.session_state.chat_history[-10:]

    st.write(ui["thinking"])

    # --- Generate Answer with Web Search + Memory ---
    response = OpenAI().responses.create(
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

    #st.write("💬 **Answer:**")
    #st.write(answer)

    # --- Text to Speech ---
    st.write(ui["speaking"])

    tts = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="verse",
        input=answer
    )

    audio_bytes = tts.read()

    st.audio(audio_bytes, format="audio/mp3")
    #st.download_button("⬇️ Download Voice", audio_bytes, "answer.mp3")

