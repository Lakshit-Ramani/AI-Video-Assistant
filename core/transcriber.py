import os
import requests
import whisper
from pydub import AudioSegment

# Sarvam REST API limit is <= 30 seconds
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# ✅ Correct endpoint for Saaras v3 and v4
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")

_model = None


def load_model():
    global _model  
    if _model is None: 
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL) 
        print("Whisper model loaded.")
    return _model 


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()  
    result = model.transcribe(chunk_path, task="transcribe")  
    return result.get("text", "")


def _send_to_sarvam(piece_path: str) -> str:
    """Send one <= 30s WAV file to Sarvam and return translated English text."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        # ✅ 'mode': 'translate' translates Indic/Hinglish directly to English
        data = {
            "model": SARVAM_MODEL,
            "mode": "translate"
        }
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Split chunk into 25-second pieces to comply with Sarvam's 30s REST limit.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    # ✅ from_file handles both .wav and .mp3 inputs safely
    audio = AudioSegment.from_file(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = "" 
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):  
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)  
        full_transcript += text + " "  

    print("Transcription complete.")
    return full_transcript.strip()