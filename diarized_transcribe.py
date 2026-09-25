import os
import argparse
import torch
import torchaudio
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Persian Medical Speaker Diarization + ASR")
    parser.add_argument("--audio", type=str, required=True, help="Input audio path (wav/mp3)")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument("--num_speakers", type=int, default=2, help="Number of speakers (default: 2)")
    parser.add_argument("--model_id", type=str, default="nezamisafa/whisper-persian-v4", help="Whisper model ID")
    parser.add_argument("--output", type=str, default="medical_transcript.txt", help="Output file path")
    return parser.parse_args()

def merge_speaker_turns(segments, max_silence=0.8):
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

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")

    # ۱. استانداردسازی فرمت فایل صوتی با ffmpeg به PCM 16kHz
    std_audio = "std_input_audio.wav"
    os.system(f'ffmpeg -y -i "{args.audio}" -ar 16000 -ac 1 -c:a pcm_s16le "{std_audio}" > /dev/null 2>&1')
    audio_path = std_audio if os.path.exists(std_audio) else args.audio

    # ۲. تفکیک گویندگان (Diarization)
    print(f"[+] Running Speaker Diarization (target speakers: {args.num_speakers})...")
    diar_pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=args.hf_token).to(device)
    diar_result = diar_pipe(audio_path, num_speakers=args.num_speakers)

    annotation = getattr(diar_result, "speaker_diarization", diar_result)
    annotation = getattr(diar_result, "annotation", annotation)

    raw_segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        raw_segments.append({"speaker": speaker, "start": round(turn.start, 2), "end": round(turn.end, 2)})

    dialogue_turns = merge_speaker_turns(raw_segments)
    print(f"[+] Identified {len(dialogue_turns)} conversation segments.")

    # ۳. بارگذاری مدل ASR پزشکی
    print(f"[+] Loading Medical ASR: {args.model_id}...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    processor = AutoProcessor.from_pretrained(args.model_id)

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=0 if torch.cuda.is_available() else -1
    )

    # تزریق پرامپت پزشکی به شکل تانسور CUDA
    med_prompt = "ویزیت دکتر، علائم بالینی، شرح حال، تجویز دارو، استامینوفن، ژلوفن، آسپرین و آزمایش."
    prompt_ids = torch.tensor(processor.get_prompt_ids(med_prompt), dtype=torch.long, device=device)

    # خواندن فایل صوتی
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    # ۴. استخراج و نگاشت دیالوگ‌ها
    print("\n" + "=" * 50)
    print("نتایج بازشناسی مکالمه بالینی:")
    print("=" * 50)

    transcript_lines = []
    for turn in dialogue_turns:
        if (turn['end'] - turn['start']) < 0.4:
            continue
            
        start_idx = int(turn['start'] * sr)
        end_idx = int(turn['end'] * sr)
        chunk = waveform[start_idx:end_idx].numpy()

        res = asr_pipeline(
            {"raw": chunk, "sampling_rate": sr},
            generate_kwargs={"language": "persian", "task": "transcribe", "prompt_ids": prompt_ids, "temperature": 0.0}
        )
        text = res["text"].strip()
        if text:
            line = f"[{turn['start']:05.2f}s -> {turn['end']:05.2f}s] {turn['speaker']}: {text}"
            print(line)
            transcript_lines.append(line)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_lines))
    print(f"\n[✓] Transcript saved to '{args.output}'")

    if os.path.exists("std_input_audio.wav"):
        os.remove("std_input_audio.wav")

if __name__ == "__main__":
    main()