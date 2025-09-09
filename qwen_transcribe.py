# ✅ Qwen2.5-VL-7B video + ASR transcript (tri-modal) pipeline — single cell
# - Installs deps (transformers, decord, qwen-vl-utils, faster-whisper)
# - Loads Qwen/Qwen2.5-VL-7B-Instruct
# - Transcribes video audio with Whisper (faster-whisper)
# - Feeds {video + transcript + prompt} into Qwen-VL
# - Provides: summarize_video_with_asr(), chunked_summarize_with_asr(), qa_over_video_with_asr()
# - Demo: reads videos from mlvu_samples/MLVU/video/9_summary/*.mp4

import os, sys, subprocess, json, math, time, glob, shutil
from pathlib import Path
import torch
import transformers
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
import decord
import numpy as np
import PIL
import qwen_vl_utils
from qwen_vl_utils import process_vision_info
from faster_whisper import WhisperModel

os.environ.setdefault("HF_HOME", str(Path.home()/".cache/huggingface"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    TORCH_DTYPE = torch.float32

print(f"[env] device={DEVICE}, dtype={TORCH_DTYPE}, model={MODEL_NAME}")

def _check_ffmpeg():
    path = shutil.which("ffmpeg")
    print(f"[ffmpeg] {'found at ' + path if path else 'NOT FOUND — please add ffmpeg to PATH (e.g., module load ffmpeg)'}", flush=True)
_check_ffmpeg()

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    min_pixels=128*16*16,
    max_pixels=256*16*16,
)
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
    device_map="auto",
    trust_remote_code=True
)


from pathlib import Path
import json

def save_asr_json(asr: dict, video_path: str) -> str:
    """
    Save ASR output as <video>.asr.json in the same folder as the video.
    Returns the JSON file path.
    """
    jpath = Path(video_path).with_suffix(".asr.json")
    obj = {
        "language": asr.get("language"),
        "duration": float(asr["segments"][-1]["end"]) if asr.get("segments") else 0.0,
        "text": asr.get("text", ""),
        "segments": asr.get("segments", []),  # list of {start, end, text}
    }
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(jpath)

from datetime import datetime

def save_summary_json(summary_text: str, video_path: str, style: str = "concise", meta: dict = None) -> str:
    """
    Save the generated summary as <video>.summary.json next to the video.
    Returns the JSON file path.
    """
    jpath = Path(video_path).with_suffix(".summary.json")
    obj = {
        "video": str(video_path),
        "summary": summary_text,
        "style": style,
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(TORCH_DTYPE).split(".")[-1],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "asr_json": Path(video_path).with_suffix(".asr.json").name,
    }
    if meta:
        obj["meta"] = meta
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(jpath)

def save_chunked_summary_json(chunks: list, final_text: str, video_path: str, meta: dict = None) -> str:
    """
    Save chunked summary as <video>.chunks.json next to the video.
    """
    jpath = Path(video_path).with_suffix(".chunks.json")
    obj = {
        "video": str(video_path),
        "final": final_text,
        "chunks": chunks,
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(TORCH_DTYPE).split(".")[-1],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "asr_json": Path(video_path).with_suffix(".asr.json").name,
    }
    if meta:
        obj["meta"] = meta
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(jpath)
def save_summary_txt(summary_text: str, video_path: str) -> str:
    """
    Save the generated summary as <video>.summary.txt next to the video.
    """
    tpath = Path(video_path).with_suffix(".summary.txt")
    with open(tpath, "w", encoding="utf-8") as f:
        f.write(summary_text.strip() + "\n")
    return str(tpath)

def _has_vision(messages):
    for m in messages:
        for it in m.get("content", []):
            if ("image" in it) or ("image_url" in it) or ("video" in it):
                return True
    return False

def _run_generation(messages, max_new_tokens=256, temperature=0.2, top_p=0.9, do_sample=False):
    image_inputs, video_inputs = (None, None)
    if _has_vision(messages):
        image_inputs, video_inputs = process_vision_info(messages)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device, dtype=TORCH_DTYPE if DEVICE == "cuda" else None)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            use_cache=True,
        )

    out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    sep = "\nassistant\n"
    return out.split(sep)[-1].strip() if sep in out else out

_ASR_MODEL_CACHE = {}

def _pick_compute_type():
    if DEVICE == "cuda":
        return "float16" 
    return "int8"        

def transcribe_video(video_path: str, asr_model_size: str = "medium", language: str = None,
                     vad: bool = True, beam_size: int = 5):
    """
    Returns:
      {"text": str, "segments": [{"start": float, "end": float, "text": str}, ...], "language": str}
    """
    compute_type = _pick_compute_type()
    key = (asr_model_size, compute_type, DEVICE)
    if key not in _ASR_MODEL_CACHE:
        print(f"[asr] loading faster-whisper model={asr_model_size}, device={DEVICE}, compute_type={compute_type}")
        _ASR_MODEL_CACHE[key] = WhisperModel(asr_model_size, device=DEVICE, compute_type=compute_type)
    model = _ASR_MODEL_CACHE[key]

    segments, info = model.transcribe(
        video_path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    full = " ".join(x["text"].strip() for x in segs).strip()
    return {"text": full, "segments": segs, "language": info.language or language}

def _ts_hhmmss_ms(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def save_srt(asr, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(asr["segments"], 1):
            f.write(f"{i}\n{_ts_hhmmss_ms(seg['start'])} --> {_ts_hhmmss_ms(seg['end'])}\n{seg['text'].strip()}\n\n")

def _duration_from_asr(asr) -> float:
    return float(asr["segments"][-1]["end"]) if asr.get("segments") else 0.0

def _asr_between(asr, start_t: float, end_t: float) -> str:
    buf = []
    for s in asr.get("segments", []):
        if (s["end"] >= start_t) and (s["start"] <= end_t):
            buf.append(s["text"].strip())
    return " ".join(buf).strip()

def _build_messages_for_video(video_path, user_text, transcript_text: str = None, max_chars: int = 6000):
    content = [{"type": "video", "video": str(video_path)}, {"type": "text", "text": user_text}]
    if transcript_text:
        tx = transcript_text if len(transcript_text) <= max_chars else (transcript_text[:max_chars] + " …[truncated]")
        content.append({
            "type": "text",
            "text": (
                "ASR TRANSCRIPT (auxiliary; prefer visual evidence if conflicts):\n" + tx
            )
        })
    return [{"role": "user", "content": content}]

def summarize_video_with_asr(video_path: str, style="concise", asr_model_size="medium",
                             language=None, max_new_tokens=256, max_transcript_chars=6000):
    asr = transcribe_video(video_path, asr_model_size=asr_model_size, language=language)

    if style == "bullet":
        user_prompt = (
            "Give a bullet-point summary of the main events, characters, and actions in this video. "
            "Use the transcript for speech/names; if transcript conflicts with visuals, prefer visuals."
        )
    elif style == "detailed":
        user_prompt = (
            "Write an exhaustive, time-ordered summary. Note shot changes, camera motions, on-screen text, "
            "who/what/where per segment, and small actions. Use bullet points + a short synopsis. "
            "Use the transcript for speech details, but prefer visuals if there is a conflict."
        )
    else:
        user_prompt = "Summarize the main events and key actions; use transcript for speech details."

    msgs = _build_messages_for_video(video_path, user_prompt, transcript_text=asr["text"], max_chars=max_transcript_chars)
    summary = _run_generation(msgs, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=False)
    return {"summary": summary, "asr": asr}

def chunked_summarize_with_asr(video_path: str, target_chunks=6, asr_model_size="medium",
                               language=None, max_new_tokens=220, max_chunk_chars=4000):
    asr = transcribe_video(video_path, asr_model_size=asr_model_size, language=language)
    total = max(_duration_from_asr(asr), 1e-6)
    edges = [i * (total / target_chunks) for i in range(target_chunks + 1)]

    chunk_summaries = []
    for i in range(target_chunks):
        t0, t1 = edges[i], edges[i+1]
        tx = _asr_between(asr, t0, t1)
        if len(tx) > max_chunk_chars:
            tx = tx[:max_chunk_chars] + " …[truncated]"
        prompt = (
            f"You are chunk {i+1}/{target_chunks}. Extract different parts of the story if possible. "
            "Write 3–6 bullet points covering distinct visual events/actions you observe. "
            "Use the transcript snippet below only for speech details."
        )
        msgs = _build_messages_for_video(video_path, prompt, transcript_text=tx, max_chars=max_chunk_chars)
        txt = _run_generation(msgs, max_new_tokens=max_new_tokens, temperature=0.3, top_p=0.9, do_sample=True)
        chunk_summaries.append(txt)

    merge_prompt = (
        "You are given partial bullet summaries from different passes of the same long video.\n"
        "1) Deduplicate overlaps.\n2) Order into a coherent timeline.\n3) Return a final bullet summary "
        "and a 1–2 sentence synopsis.\n\n"
        + "\n\n".join([f"--- Chunk {i+1} ---\n{cs}" for i, cs in enumerate(chunk_summaries)])
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": merge_prompt}]}]
    final = _run_generation(messages, max_new_tokens=280, temperature=0.2, do_sample=False)
    return {"chunks": chunk_summaries, "final": final, "asr": asr}

def qa_over_video_with_asr(video_path: str, question: str, asr_model_size="medium", language=None,
                           max_new_tokens=256, max_transcript_chars=6000):
    asr = transcribe_video(video_path, asr_model_size=asr_model_size, language=language)
    user_prompt = (
        "Answer this question about the video based on what is visually present. "
        "Use the transcript only for spoken names/words; if conflicts, prefer visuals.\n\n"
        f"Question: {question}"
    )
    msgs = _build_messages_for_video(video_path, user_prompt, transcript_text=asr["text"], max_chars=max_transcript_chars)
    answer = _run_generation(msgs, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=False)
    return {"answer": answer, "asr": asr}

SAMPLES_DIR = Path("mlvu_samples/MLVU/video/9_summary")
videos = sorted(glob.glob(str(SAMPLES_DIR / "*.mp4")))
print(f"[demo] found {len(videos)} videos under {SAMPLES_DIR}")

if videos:
    demo_video = videos[0]
    print(f"[demo] summarizing with ASR: {demo_video}")
    print("—"*80)
    out = summarize_video_with_asr(demo_video, style="detailed", asr_model_size="medium", max_new_tokens=5000)
    print(out["summary"])
    summary_txt_path = save_summary_txt(out["summary"], demo_video)
    print(f"[summary] TXT saved at: {summary_txt_path}")
    print("\n[ASR full transcript]\n", out["asr"]["text"])
    json_path = save_asr_json(out["asr"], demo_video)
    print(f"[asr] JSON saved at: {json_path}")
    summary_json_path = save_summary_json(out["summary"], demo_video, style="detailed")

    srt_path = str(Path(demo_video).with_suffix(".srt"))
    save_srt(out["asr"], srt_path)
    print(f"\n[asr] language={out['asr']['language']}, segments={len(out['asr']['segments'])}, srt={srt_path}")

    qa = qa_over_video_with_asr(demo_video, "Who appears first and what are they doing?")
    print("\n[QA] ", qa["answer"])
else:
    print("[demo] No videos found. Put some .mp4 files in mlvu_samples/")
