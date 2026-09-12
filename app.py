"""
Persian Medical Speech Recognition - Interactive Web UI
Built with Gradio & Whisper Large-v3 (LoRA adapted)
"""

import os
import torch
import torchaudio
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

BASE_MODEL_ID = "nezamisafa/whisper-persian-v4"
ADAPTER_PATH = "./medical_adapter"

# انتخاب خودکار کارت گرافیک یا پردازنده
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"[+] Loading models on {device.upper()}...")

processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=compute_dtype,
    low_cpu_mem_usage=True
).to(device)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("[+] Model loaded successfully. Ready for inference.")


def transcribe_live(audio_path):
    if not audio_path:
        return "لطفاً ابتدا صدایی با میکروفون ضبط کرده یا فایل صوتی آپلود نمایید."

    # آماده‌سازی و بازنمونه‌برداری صوت به ۱۶ کیلوهرتز
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    # استخراج ویژگی‌ها و رونویسی
    inputs = processor(wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=compute_dtype)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="fa", task="transcribe")

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


# طراحی رابط کاربری وب
demo = gr.Interface(
    fn=transcribe_live,
    inputs=gr.Audio(
        sources=["microphone", "upload"], 
        type="filepath", 
        label="ضبط زنده با میکروفون یا آپلود فایل صوتی"
    ),
    outputs=gr.Textbox(
        label="متن تشخیص‌داده‌شده بالینی (خروجی مدل)", 
        rtl=True,
        lines=4
    ),
    title="سیستم تبدیل صوت به متن پزشکی (Persian Medical Whisper ASR)",
    description="برای شروع روی دکمه ضبط کلیک کرده، جمله‌ای مرتبط با شرح‌حال یا علائم پزشکی بگویید و کلید Submit را بزنید.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)