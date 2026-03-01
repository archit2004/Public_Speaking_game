
import os
import tempfile
import threading
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from TTS.api import TTS
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
_playback_queue = queue.Queue()
_playback_worker_started = False
_playback_worker_lock = threading.Lock()
play_lock = threading.Lock()      
is_speaking = threading.Event()   

def _write_bytes_to_tempfile(data: bytes, suffix=".webm") -> str:

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            os.remove(tmp.name)
        except Exception:
            pass
        raise


def _playback_worker():
    while True:
        item = _playback_queue.get()
        if item is None:
            break
        fname, cleanup = item
        try:
            is_speaking.set()
            data, sr = sf.read(fname, dtype="float32")
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            print("Playback worker error:", e)
        finally:
            is_speaking.clear()
            if cleanup:
                try:
                    os.remove(fname)
                except Exception:
                    pass
        _playback_queue.task_done()

def _ensure_playback_worker():
    global _playback_worker_started
    if not _playback_worker_started:
        with _playback_worker_lock:
            if not _playback_worker_started:
                t = threading.Thread(target=_playback_worker, daemon=True)
                t.start()
                _playback_worker_started = True

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
whisper_model = whisper.load_model(WHISPER_MODEL, device="cuda")

TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC_ph")
tts_engine = TTS(TTS_MODEL)
tts_engine.to("cuda")
def _write_bytes_to_tempfile(data: bytes, suffix=".wav") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def speech_to_text(file_or_path) -> str:
    if not file_or_path:
        return ""

    tmp_path = None
    try:
        if isinstance(file_or_path, (bytes, bytearray)):
            tmp_path = _write_bytes_to_tempfile(file_or_path, suffix=".webm")
            path = tmp_path
        elif hasattr(file_or_path, "read"):
            data = file_or_path.read()
            if isinstance(data, (bytes, bytearray)):
                tmp_path = _write_bytes_to_tempfile(data, suffix=".webm")
                path = tmp_path
            else:
                raise ValueError("file-like read() did not return bytes")
        elif isinstance(file_or_path, str) and os.path.exists(file_or_path):
            path = file_or_path
        else:
            raise ValueError("speech_to_text expects bytes, file-like object, or existing file path")

        print(f"[speech_to_text] Passing file to whisper: {path}")
        result = whisper_model.transcribe(path)
        text = result.get("text", "").strip()
        print("[speech_to_text] Transcription:", text)
        return text

    except Exception as e:
        print("Error in speech_to_text:", e)
        print(traceback.format_exc())
        return ""
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
def tts_enqueue_chunk(text_chunk: str, *, samplerate=22050, cleanup=True) -> str:
    if not text_chunk or not text_chunk.strip():
        return ""

    _ensure_playback_worker()
    fname = os.path.join(tempfile.gettempdir(), f"tts_stream_{int(time.time()*1000)}.wav")
    try:
        tts_engine.tts_to_file(text=text_chunk, file_path=fname)
    except Exception as e:
        print("TTS synth error:", e)
        return ""
    _playback_queue.put((fname, cleanup))
    return fname

def text_to_speech(text: str, samplerate=22050) -> str:
    if not text:
        return ""
    fname = os.path.join(tempfile.gettempdir(), f"tts_{int(time.time()*1000)}.wav")
    try:
        tts_engine.tts_to_file(text=text, file_path=fname)
    except Exception as e:
        print("TTS synth error:", e)
        return ""
    _ensure_playback_worker()
    _playback_queue.put((fname, True))
    return fname

def record_audio(silence_threshold=0.1, silence_duration=3.0, samplerate=16000) -> str:
    while is_speaking.is_set():
        time.sleep(0.1)

    q = queue.Queue()
    audio_data = []
    silent_chunks = 0
    silence_limit = int(silence_duration * samplerate / 1024)
    max_chunks = int(30 * samplerate / 1024)  

    def callback(indata, frames, time, status):
        if status:
            print("InputStream status:", status)
        q.put(indata.copy())

    print("Speak now... (auto-stop after silence)")
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, blocksize=1024):
        chunk_count = 0
        while chunk_count < max_chunks:
            chunk = q.get()
            audio_data.append(chunk)
            rms = np.sqrt(np.mean(chunk**2))
            if rms < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
            if silent_chunks >= silence_limit:
                break
            chunk_count += 1

    audio = np.concatenate(audio_data, axis=0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio, samplerate)
    return tmp.name

def clarity_score(audio_path: str, transcript: str):
    """
    Analyze clarity of the user's speech using audio + transcript.
    Returns a dict with a numeric score and feedback.
    """
    try:
        audio, samplerate = sf.read(audio_path)
    except Exception as e:
        return {"score": 0, "feedback": f"Audio error: {e}"}


    rms = np.sqrt(np.mean(audio**2))
    loudness_db = 20 * np.log10(rms + 1e-6)  

    duration_sec = len(audio) / samplerate


    words = transcript.strip().split()
    num_words = len(words)
    wpm = (num_words / duration_sec) * 60 if duration_sec > 0 else 0


    fillers = re.findall(r"\b(um+|uh+|like|you know)\b", transcript.lower())
    filler_count = len(fillers)



    score = 100


    if loudness_db < -30:
        score -= 20
        loudness_feedback = "Your voice is too soft."
    elif loudness_db > -5:
        score -= 10
        loudness_feedback = "Your voice is a bit too loud."
    else:
        loudness_feedback = "Good voice volume."


    if wpm < 100:
        score -= 15
        speed_feedback = "You spoke too slowly."
    elif wpm > 190:
        score -= 15
        speed_feedback = "You spoke too quickly."
    else:
        speed_feedback = "Good speaking pace."


    if filler_count > 3:
        score -= filler_count * 2
        filler_feedback = f"You used filler words {filler_count} times."
    else:
        filler_feedback = "Minimal filler words used."

    score = max(0, score) 

    feedback = f"{loudness_feedback} {speed_feedback} {filler_feedback}"

    return {"score": score, "feedback": feedback}


def speak(text: str, samplerate: int = 22050, delete_after: bool = True) -> str:
    """
    Convert `text` to speech using your tts_engine,
    play it once, and optionally delete the temp file.

    Returns the path to the generated WAV.
    """
    if not text or not text.strip():
        raise ValueError("Text must be non-empty")

    wav_path = os.path.join(
        tempfile.gettempdir(), f"tts_{int(time.time()*1000)}.wav"
    )

    tts_engine.tts_to_file(text=text, file_path=wav_path)

    data, sr = sf.read(wav_path, dtype="float32")
    sd.play(data, sr)
    sd.wait()  

    if delete_after:
        os.remove(wav_path)

    return wav_path

