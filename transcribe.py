"""
Persian Medical Speech-to-Text (ASR)
Inference script using Whisper Large-v3 with a domain-adapted LoRA checkpoint.
"""

import os
import sys
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# Configuration
BASE_MODEL_ID = "nezamisafa/whisper-persian-v4"
ADAPTER_PATH = "./medical_adapter"

# Hardware target selection
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"[+] Initializing inference engine on target hardware: {device.upper()}")

# 1. Load base tokenizer and processor
processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

# 2. Load the base Whisper Large-v3 architecture
base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=compute_dtype,
    low_cpu_mem_usage=True
).to(device)

# 3. Mount fine-tuned medical LoRA adapter weights
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("[+] Model and domain adapter successfully loaded.")


def transcribe_audio(audio_path: str) -> str:
    """
    Preprocess input audio to 16kHz mono, extract log-Mel features,
    and generate Persian medical transcription.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Target audio file '{audio_path}' does not exist.")

    # Load audio waveform
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono channel if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16,000 Hz required by Whisper
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Extract log-Mel spectrogram features
    audio_array = waveform.squeeze(0).numpy()
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=compute_dtype)

    # Run autoregressive text generation constrained to Persian
    with torch.no_grad():
        generated_token_ids = model.generate(
            input_features,
            language="fa",
            task="transcribe"
        )

    # Decode token IDs into clean human-readable text
    transcription = processor.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
    return transcription


if __name__ == "__main__":
    # Select audio file via CLI argument or default to local sample
    target_file = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"

    print(f"[+] Processing audio: {target_file}")
    result_text = transcribe_audio(target_file)

    print("\n" + "=" * 50)
    print("TRANSCRIPTION OUTPUT:")
    print("=" * 50)
    print(result_text)
    print("=" * 50)