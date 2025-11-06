# streamlit_kids_ai_app.py
# Complete Streamlit app for a kid-friendly voice assistant with:
# - Microphone input (browser) using streamlit-audiorecorder
# - Whisper transcription via OpenAI
# - Web search (DuckDuckGo) for grounding answers
# - LLM reasoning (OpenAI chat completion)
# - Text-to-Speech (OpenAI TTS if available, fallback to gTTS)
# - Multilanguage support (speech-in, response language, TTS)
# - Simple content moderation

import os
import tempfile
import json
import time
from pathlib import Path
from typing import Optional

import streamlit as st
from duckduckgo_search import DDGS
from pydub import AudioSegment

# Browser mic recorder (pip install streamlit-audiorecorder)
try:
    from streamlit_audiorecorder import audiorecorder
except Exception as e:
    audiorecorder = None

# OpenAI client - try to support both `openai` and `OpenAI` client styles
try:
    # Preferred: new OpenAI Python client
    from openai import OpenAI
    openai_client = OpenAI
    NEW_OPENAI = True
except Exception:
    import openai
    NEW_OPENAI = False


# ---------------------------
# Helper utilities
# ---------------------------

def load_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("Please set the OPENAI_API_KEY environment variable before running the app.")
        st.stop()
    return key


def save_bytes_to_file(b: bytes, suffix: str = ".wav") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(b)
    tmp.flush()
    tmp.close()
    return tmp.name


def convert_to_mp3(in_path: str) -> str:
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    audio = AudioSegment.from_file(in_path)
    audio.export(out.name, format="mp3")
    return out.name


# ---------------------------
# Transcription (Whisper)
# ---------------------------

def transcribe_with_openai(in_audio_path: str, language: Optional[str] = None) -> str:
    """Transcribe audio using OpenAI Whisper-like endpoint.
    This tries to use the newer OpenAI client if available, otherwise falls back.
    """
    if NEW_OPENAI:
        client = openai_client(api_key=os.getenv("OPENAI_API_KEY"))
        # new client might expose audio.transcriptions.create
        with open(in_audio_path, "rb") as af:
            # model name here is an example; update if needed in your environment
            resp = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=af, language=language)
            # response shape may vary by client version
            return resp.text
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        with open(in_audio_path, "rb") as af:
            # many older examples use openai.Audio.transcribe or openai.Transcription.create
            try:
                resp = openai.Audio.transcribe("gpt-4o-mini-transcribe", af, language=language)
                return resp["text"] if isinstance(resp, dict) else resp.text
            except Exception:
                # fallback to whisper-1 name if available
                resp = openai.Audio.transcribe("whisper-1", af, language=language)
                return resp.get("text") if isinstance(resp, dict) else resp.text


# ---------------------------
# Moderation
# ---------------------------

def moderate_text(text: str) -> dict:
    """Return moderation result dict indicating whether flagged."""
    if NEW_OPENAI:
        client = openai_client(api_key=os.getenv("OPENAI_API_KEY"))
        res = client.moderations.create(input=text)
        return res
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        try:
            res = openai.Moderation.create(input=text)
            return res
        except Exception:
            # best-effort: return empty safe result
            return {"results": [{"flagged": False}]}


# ---------------------------
# Web search (DuckDuckGo)
# ---------------------------

def quick_search(question: str, max_results: int = 3) -> str:
    ddgs = DDGS()
    snippets = []
    for idx, r in enumerate(ddgs.text(question, max_results=max_results)):
        # r is dict-like with keys 'title' and 'body'
        title = r.get("title") or ""
        body = r.get("body") or ""
        snippets.append(f"- {title}: {body[:400]}")
        if idx + 1 >= max_results:
            break
    return "\n".join(snippets)


# ---------------------------
# LLM Answer Generation
# ---------------------------

def generate_answer(question: str, context: str, language: str, child_age: int = 8) -> str:
    sys_prompt = (
        "You are a friendly, patient tutor for children. Answer simply and positively. "
        f"Use short sentences suitable for a child of about {child_age} years old. "
        "If you don't know the answer, say you don't know and explain how you might find out."
    )
    # Ask the model to respond in the requested language
    user_content = (
        f"Language: {language}\nQuestion: {question}\n\nContext (from web search):\n{context}\n\n"
        "Provide a short concise answer appropriate for a child, and one follow-up question to keep them curious."
    )

    if NEW_OPENAI:
        client = openai_client(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        # shape may vary; try to extract text
        try:
            return resp.choices[0].message.content
        except Exception:
            return getattr(resp.choices[0], "message", {}).get("content", str(resp))
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return resp["choices"][0]["message"]["content"]


# ---------------------------
# Text-to-Speech
# ---------------------------

def tts_openai_speech(answer_text: str, out_path: str, language: str = "en") -> Optional[str]:
    """Try OpenAI TTS (if available). Falls back to None so caller can use gTTS.
    The exact API for TTS in OpenAI Python client varies by version. This function
    does a best-effort attempt and will return the path to the saved audio file.
    """
    try:
        if NEW_OPENAI:
            client = openai_client(api_key=os.getenv("OPENAI_API_KEY"))
            # Example: client.audio.speech.create(...)
            with open(out_path, "wb") as fout:
                resp = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=answer_text, language=language)
                # resp may be bytes or a stream
                if hasattr(resp, "read"):
                    fout.write(resp.read())
                elif isinstance(resp, (bytes, bytearray)):
                    fout.write(resp)
                else:
                    # try to get data
                    data = getattr(resp, "data", None) or getattr(resp, "content", None)
                    if isinstance(data, (bytes, bytearray)):
                        fout.write(data)
                    else:
                        fout.write(str(resp).encode("utf-8"))
            return out_path
        else:
            # older client doesn't have TTS; return None so fallback will run
            return None
    except Exception:
        return None


def tts_gtts(answer_text: str, out_path: str, language: str = "en") -> str:
    # Fallback: gTTS (pip install gTTS)
    try:
        from gtts import gTTS
    except Exception:
        raise RuntimeError("gTTS not installed. Install gTTS or provide OpenAI TTS access.")
    tts = gTTS(text=answer_text, lang=language)
    tts.save(out_path)
    return out_path


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Kids Voice AI", layout="centered")
st.title("🎧 Kids Voice AI — Talk and Learn")

# Instructions and sidebar
st.sidebar.header("Settings")
supported_languages = {
    "English": "en",
    "Lithuanian": "lt",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Russian": "ru",
    "Polish": "pl",
}
lang_name = st.sidebar.selectbox("Response & TTS Language", list(supported_languages.keys()), index=0)
lang_code = supported_languages[lang_name]
child_age = st.sidebar.slider("Target child age", 4, 14, 8)
max_search_results = st.sidebar.slider("Search results to use", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("Make sure OPENAI_API_KEY is set in your environment.")

# API key check
load_api_key()

col1, col2 = st.columns([3, 1])
with col1:
    st.write("Press the record button, ask a question out loud, then stop the recording.")
    if audiorecorder is None:
        st.warning("`streamlit-audiorecorder` is not installed. Use file upload below or install it: `pip install streamlit-audiorecorder`.")

    audio_bytes = None
    if audiorecorder is not None:
        audio_bytes = audiorecorder(label="Hold to record your question", text="Recording...", icon="microphone")

    st.write("Or upload a recorded audio file (wav/mp3/m4a):")
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"], accept_multiple_files=False)

with col2:
    st.write(" ")
    if st.button("Clear"):
        st.experimental_rerun()

# Choose source: recorded bytes > uploaded file
input_audio_path = None
if audio_bytes:
    # audiorecorder gives bytes-like or a file-like object depending on version
    if isinstance(audio_bytes, (bytes, bytearray)):
        input_audio_path = save_bytes_to_file(audio_bytes, suffix=".wav")
    else:
        # sometimes returns numpy array or other; try to handle
        try:
            b = audio_bytes.tobytes()
            input_audio_path = save_bytes_to_file(b, suffix=".wav")
        except Exception:
            input_audio_path = None

if uploaded and input_audio_path is None:
    input_audio_path = save_bytes_to_file(uploaded.read(), suffix=Path(uploaded.name).suffix)

if input_audio_path:
    st.success("Audio received — transcribing...")
    with st.spinner("Transcribing (Whisper)..."):
        try:
            transcription = transcribe_with_openai(input_audio_path, language=lang_code)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            transcription = ""

    if transcription:
        st.markdown("**Child said:**")
        st.write(transcription)

        # Web search for context
        with st.spinner("Searching the web for facts..."):
            try:
                context = quick_search(transcription, max_results=max_search_results)
            except Exception as e:
                st.warning(f"Search failed or returned no results: {e}")
                context = ""

        # Generate answer
        with st.spinner("Generating a friendly answer..."):
            try:
                answer = generate_answer(transcription, context, language=lang_code, child_age=child_age)
            except Exception as e:
                st.error(f"LLM generation failed: {e}")
                answer = "Sorry, I couldn't come up with an answer right now."

        # Moderation (check answer before speaking)
        try:
            mod = moderate_text(answer)
            flagged = False
            if isinstance(mod, dict):
                flagged = mod.get("results", [{}])[0].get("flagged", False)
            else:
                # new client object may have attribute results
                flagged = getattr(mod, "results", [None])[0].get("flagged", False)
        except Exception:
            flagged = False

        if flagged:
            st.warning("The generated answer was flagged by moderation. Showing a safe fallback message.")
            answer_for_display = "I'm sorry — I can't answer that. Let's ask something else!"
        else:
            answer_for_display = answer

        st.markdown("**Assistant (text):**")
        st.write(answer_for_display)

        # Text-to-speech
        out_mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts_done = False
        with st.spinner("Converting text to speech..."):
            # Try OpenAI TTS first
            outp = tts_openai_speech(answer_for_display, out_mp3_path, language=lang_code)
            if outp:
                tts_done = True
            else:
                # fallback to gTTS
                try:
                    outp = tts_gtts(answer_for_display, out_mp3_path, language=lang_code)
                    tts_done = True
                except Exception as e:
                    st.error(f"TTS failed: {e}")
                    tts_done = False

        if tts_done and Path(out_mp3_path).exists():
            audio_bytes_out = open(out_mp3_path, "rb").read()
            st.audio(audio_bytes_out, format="audio/mp3")
            st.success("Played answer — you can replay using the player above.")
        else:
            st.info("Text-to-speech unavailable. Showing text answer only.")

        # Offer the mp3 for download
        try:
            st.download_button("Download spoken answer (mp3)", data=open(out_mp3_path, "rb"), file_name="answer.mp3", mime="audio/mpeg")
        except Exception:
            pass

        # Clean up temp audio input if created
        try:
            os.remove(input_audio_path)
        except Exception:
            pass

else:
    st.info("Record a question or upload an audio file to start.")


# Footer notes
st.markdown("---")
st.caption("This demo uses OpenAI APIs (transcription, chat, moderation, and optionally TTS)." )
st.caption("Make sure you have a valid OPENAI_API_KEY set as an environment variable before running the app.")


# ---------------------------
# End of file
# ---------------------------
