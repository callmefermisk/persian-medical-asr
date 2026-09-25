import os
import base64
import torch
import torchaudio
import gradio as gr
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from peft import PeftModel

# Hardware acceleration setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Pre-configured authentication token for headless execution
DEFAULT_TOKEN_B64 = "aGZfa1JJRmRKanNVVG9UU05Ma1JMR25NQkdaVFJjVGpoQmJnUg=="
HF_TOKEN = os.environ.get("HF_TOKEN") or base64.b64decode(DEFAULT_TOKEN_B64).decode("utf-8")

print(f"[+] Initializing models on target device: {device}")

# 1. Load neural speaker diarization pipeline
diar_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
).to(device)

# 2. Load base Whisper model and mount medical LoRA adapter if present
base_model_id = "nezamisafa/whisper-persian-v4"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    base_model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True
).to(device)

if os.path.exists("medical_adapter"):
    print("[+] Detected medical LoRA adapter. Mounting fine-tuned weights...")
    model = PeftModel.from_pretrained(model, "medical_adapter")

processor = AutoProcessor.from_pretrained(base_model_id)
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=0 if torch.cuda.is_available() else -1
)

# Domain-specific prompt biasing to steer token probabilities toward medical lexicon
med_prompt = "ویزیت دکتر، علائم بالینی، شرح حال، تجویز دارو، استامینوفن، ژلوفن، آسپرین و آزمایش."
prompt_ids = torch.tensor(processor.get_prompt_ids(med_prompt), dtype=torch.long, device=device)

def merge_speaker_turns(segments, max_silence=0.8):
    """
    Merges consecutive audio intervals belonging to the same speaker
    if the intervening pause is below the threshold, preventing context fragmentation.
    """
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        if seg['speaker'] == last['speaker'] and (seg['start'] - last['end']) <= max_silence:
            last['end'] = max(last['end'], seg['end'])
        else:
            merged.append(seg.copy())
    return merged

def process_clinical_audio(audio_path, speaker_mode):
    """
    Handles audio standardization, diarization inference, dynamic turn alignment,
    and clinical Persian transcription.
    """
    if not audio_path:
        return "⚠️ لطفاً ابتدا یک فایل صوتی آپلود کنید یا صدایتان را ضبط نمایید."

    # Standardize input waveform to 16 kHz mono PCM WAV via ffmpeg
    std_audio = "temp_gradio_audio.wav"
    os.system(f'ffmpeg -y -i "{audio_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{std_audio}" > /dev/null 2>&1')
    active_audio = std_audio if os.path.exists(std_audio) else audio_path

    # Configure diarization constraints based on selected UI mode
    num_spk = None
    if speaker_mode == "تک‌گوینده (دیکته پزشک)":
        num_spk = 1
    elif speaker_mode == "دو گوینده (پزشک و بیمار)":
        num_spk = 2

    # Perform acoustic diarization
    if num_spk:
        diar_result = diar_pipeline(active_audio, num_speakers=num_spk)
    else:
        # Automatic speaker clustering between 1 and 5 participants
        diar_result = diar_pipeline(active_audio, min_speakers=1, max_speakers=5)

    annotation = getattr(diar_result, "speaker_diarization", diar_result)
    annotation = getattr(diar_result, "annotation", annotation)

    raw_segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        raw_segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })

    unique_speakers = list(set([s["speaker"] for s in raw_segments]))
    dialogue_turns = merge_speaker_turns(raw_segments)

    # Load audio array for interval slicing
    waveform, sr = torchaudio.load(active_audio)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    # Route 1: Single-speaker clinical memo formatting
    if len(unique_speakers) <= 1:
        res = asr_pipeline(
            active_audio,
            generate_kwargs={
                "language": "persian",
                "task": "transcribe",
                "prompt_ids": prompt_ids,
                "temperature": 0.0
            }
        )
        output_text = "### 📋 گزارش بالینی (تک‌گوینده):\n\n"
        output_text += f"> {res['text'].strip()}"
        return output_text

    # Route 2: Multi-speaker doctor-patient dialogue attribution
    output_text = f"### 🩺 مکالمه بالینی تفکیک‌شده (تشخیص خودکار: {len(unique_speakers)} گوینده)\n\n"
    for turn in dialogue_turns:
        # Ignore spurious noise artifacts below 0.4 seconds
        if (turn['end'] - turn['start']) < 0.4:
            continue
            
        start_idx = int(turn['start'] * sr)
        end_idx = int(turn['end'] * sr)
        chunk = waveform[start_idx:end_idx].numpy()

        res = asr_pipeline(
            {"raw": chunk, "sampling_rate": sr},
            generate_kwargs={
                "language": "persian",
                "task": "transcribe",
                "prompt_ids": prompt_ids,
                "temperature": 0.0
            }
        )
        text = res["text"].strip()
        if text:
            output_text += f"**[{turn['start']:05.2f}s ➔ {turn['end']:05.2f}s] {turn['speaker']}**:\n{text}\n\n---\n"

    return output_text

# Construct Gradio application interface
custom_theme = gr.themes.Soft(primary_hue="teal")

with gr.Blocks(theme=custom_theme, title="Persian Medical ASR") as demo:
    gr.Markdown(
        """
        # 🏥 سامانه هوشمند بازشناسی گفتار بالینی و تفکیک دیالوگ
        **مدل پایه:** Whisper Large-v3 بهینه‌شده با LoRA پزشکی | **موتور تفکیک:** PyAnnote 3.1
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="ورودی صوت (ضبط مستقیم یا آپلود فایل)"
            )
            mode_selector = gr.Radio(
                choices=["تشخیص خودکار (Auto-Detect)", "دو گوینده (پزشک و بیمار)", "تک‌گوینده (دیکته پزشک)"],
                value="تشخیص خودکار (Auto-Detect)",
                label="حالت پردازش گویندگان"
            )
            submit_btn = gr.Button("🔍 پردازش و پیاده‌سازی متن", variant="primary")

        with gr.Column(scale=1):
            output_display = gr.Markdown(label="متن استخراج‌شده")

    submit_btn.click(
        fn=process_clinical_audio,
        inputs=[audio_input, mode_selector],
        outputs=output_display
    )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=False)