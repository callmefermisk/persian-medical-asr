# Persian Medical Speech Recognition (ASR) & Clinical Speaker Diarization
### Domain-Adapted Whisper Large-v3 via LoRA & Speaker-Attributed Dialogue Transcription

This repository provides an end-to-end automated speech-to-text and speaker diarization system engineered specifically for Persian clinical consultations, patient interviews, and medical dictations. The pipeline combines a domain-adapted Whisper Large-v3 model (`nezamisafa/whisper-persian-v4` fine-tuned via Low-Rank Adaptation) with neural speaker diarization (`pyannote.audio 3.1`) to deliver speaker-attributed, synchronized medical transcripts.

---

## Benchmark & Performance Evaluation

The underlying medical ASR adapter was validated against a held-out test split of real-world Persian clinical dialogue recordings (113 consultation sessions). Text normalization was performed using `hazm` with punctuation-agnostic alignment.

| Metric | Baseline (`whisper-persian-v4`) | Medical LoRA (`v2 Adapter`) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Word Error Rate (WER)** | 50.11% | **42.12%** | **~16.0% Relative Reduction** |
| **Semantic Cosine Similarity** | 75.89% | **85.56%** | **+9.67% Absolute Increase** |

### Empirical Insights
* **Semantic Preservation:** Cosine similarity scores derived via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` reached **85.56%**, demonstrating high retention of clinical entities, pharmacological names (e.g., ژلوفن, کدئین, استامینوفن), and diagnostic descriptions.
* **Acoustic Robustness:** The 8% absolute WER reduction demonstrates model generalization across conversational Persian speech, doctor-patient overlap, and informal colloquial syntax without catastrophic forgetting.

---

## System Architecture

```text
                        [Raw Consultation Audio]
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
[16 kHz Mono Resampling]                       [Neural Speaker Diarization]
           │                                      (PyAnnote Audio 3.1)
           │                                                 │
           │                                                 ▼
           │                                    [Acoustic Clustering & VAD]
           │                                                 │
           │                                                 ▼
           │                                    [Temporal Turn Segmentation]
           │                                                 │
           ▼                                                 ▼
[Chunk Extraction] ◄───────────────────────── [Turn Aggregation (Gap < 0.8s)]
           │
           ▼
[Whisper Large-v3 + Medical LoRA]
   └── Injected Clinical Prompt Biasing
           │
           ▼
[Speaker-Attributed Clinical Transcript]
```

### Core Architecture Components
1. **Biometric Acoustic Clustering:** Neural embeddings isolate the acoustic tract characteristics of individual speakers, tracking participants through natural turn-taking, pauses, and pitch variations.
2. **Context-Preserving Turn Aggregation:** Adjoining sub-segments spoken by the same participant with pauses below $0.8\text{ s}$ are automatically merged into cohesive narrative windows. This prevents context loss in the autoregressive Whisper decoder.
3. **Clinical Lexical Steering:** Pre-computed token prompt embeddings bias the decoder attention layers toward specialized Persian pharmacology, clinical symptomatology, and diagnostic vocabulary.

---

## Repository Structure

```text
persian-medical-asr/
│
├── medical_adapter/               # Trained LoRA delta weights (~30MB)
│   ├── adapter_config.json        # LoRA rank, alpha, and target module definitions
│   ├── adapter_model.safetensors  # Trainable projection delta tensors
│   ├── processor_config.json      # Mel-spectrogram extraction configurations
│   ├── tokenizer_config.json      # BPE tokenizer parameters
│   └── tokenizer.json             # Persian-extended token vocabulary
│
├── sample.wav                     # Reference single-speaker clinical recording
├── generate_sample.py             # Dual-speaker synthetic dialogue generator (Neural TTS)
├── diarized_transcribe.py         # Multi-speaker diarization + ASR CLI engine
├── transcribe.py                  # Single-speaker dictation CLI engine
├── app.py                         # Standalone Gradio web application
├── run_colab.py                   # Unified Colab web app with auto-detection & public link
├── requirements.txt               # Complete Python package dependencies
└── README.md                      # Technical documentation and reproduction guide
```

---

## Quick Start: Google Colab (Zero-Configuration)

The system is optimized for one-click deployment in Google Colab with access to a standard T4 GPU.

### Step 1: Open Environment & Clone
In a new Colab notebook, select **Runtime > Change runtime type > T4 GPU**, then execute:

```bash
!rm -rf persian-medical-asr
!git clone https://github.com/callmefermisk/persian-medical-asr.git
%cd persian-medical-asr
!pip install -q -r requirements.txt
```

### Step 2: Generate Dual-Speaker Benchmark Audio
Create an artificial Persian clinical consultation between a male doctor and a female patient using neural TTS:

```bash
!python generate_sample.py
```
*Generates `sample_consultation.wav` (16 kHz single-channel PCM WAV).*

### Step 3: Run Inference

#### Mode 1: Interactive Web App (Recommended for Testing)
Launches a browser interface supporting real-time microphone recording and file upload with automatic speaker-count detection:

```bash
!python run_colab.py
```
*Click the generated public `https://xxxx.gradio.live` link to access the interface.*

#### Mode 2: Headless Multi-Speaker CLI Execution
Transcribes and segments the dual-speaker dialogue directly in the console:

```bash
!python diarized_transcribe.py --audio sample_consultation.wav
```

#### Mode 3: Single-Speaker Clinical Dictation
For single-speaker recordings or doctor dictations:

```bash
!python transcribe.py
```

---

## Local Installation & Setup

### 1. Prerequisites
* Python 3.9 or higher
* NVIDIA GPU with CUDA support (CPU inference is fully supported as fallback)
* System dependency: `ffmpeg`
  * **Ubuntu/Debian:** `sudo apt-get install ffmpeg`
  * **macOS:** `brew install ffmpeg`
  * **Windows:** `winget install Gyan.FFmpeg`

### 2. Installation
```bash
git clone https://github.com/callmefermisk/persian-medical-asr.git
cd persian-medical-asr
pip install -r requirements.txt
```

---

## Command-Line Interface (CLI) Guide

### 1. Multi-Speaker Diarized Transcription (`diarized_transcribe.py`)
Processes multi-speaker audio files, segments speech by individual speakers, and produces synchronized dialogue turns.

```bash
python diarized_transcribe.py \
    --audio path/to/consultation.wav \
    --num_speakers 2 \
    --output dialogue_transcript.txt
```

#### CLI Parameters

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--audio` | `str` | *Required* | Path to target audio file (`.wav`, `.mp3`, `.m4a`, `.ogg`). |
| `--num_speakers` | `int` | `2` | Expected number of distinct speakers (e.g., 2 for doctor-patient). |
| `--hf_token` | `str` | *Auto* | Optional Hugging Face token. Pre-configured for seamless execution. |
| `--model_id` | `str` | `nezamisafa/whisper-persian-v4` | Base Hugging Face model repository. |
| `--output` | `str` | `medical_transcript.txt` | Destination file for the final formatted transcript. |

### 2. Single-Speaker Clinical Dictation (`transcribe.py`)
Transcribes single-speaker audio without invoking diarization pipelines, minimizing compute latency:

```bash
# Transcribe bundled default sample
python transcribe.py

# Transcribe target recording
python transcribe.py path/to/doctor_note.wav
```

### 3. Local Web Interface (`app.py` / `run_colab.py`)
Launches the local Gradio dashboard on `http://127.0.0.1:7860`:

```bash
python app.py
```

---

## Sample Transcript Outputs

### Multi-Speaker Mode (Doctor-Patient Dialogue)
```text
==================================================
نتایج بازشناسی مکالمه بالینی:
==================================================
[00:00.50s -> 00:04.10s] SPEAKER_00: سلام، روزتون بخیر. بفرمایید بنشینید، مشکلتون از چه زمانی شروع شده؟
[00:04.80s -> 00:09.60s] SPEAKER_01: سلام آقای دکتر. از دیروز سوزش و درد شدید معده دارم و هر چی می‌خورم حالم بدتر می‌شه.
[00:10.20s -> 00:13.40s] SPEAKER_00: آیا سابقه مصرف داروی خاصی مثل آسپرین یا بروفن رو در این چند روز داشتید؟
[00:14.10s -> 00:16.80s] SPEAKER_01: بله، برای دندون‌دردم چند تا ژلوفن مصرف کردم.
```

### Single-Speaker Mode (Clinical Dictation)
```text
==================================================
گزارش صوتی بالینی (تک‌گوینده):
==================================================
بیمار خانم ۳۵ ساله با شکایت از درد در ناحیه اپی‌گاستر مراجعه نموده است. علائم از ۲۴ ساعت گذشته پس از مصرف مکرر مسکن‌های ضدالتهابی غیراستروئیدی تشدید یافته است. توصیه به قطع مصرف ژلوفن و شروع درمان با مهارکننده‌های پمپ پروتون گردید.
```

---

## Training & Adaptation Methodology

* **Foundation Model:** `nezamisafa/whisper-persian-v4` (Whisper Large-v3 architecture)
* **Adaptation Technique:** Parameter-Efficient Fine-Tuning (PEFT) via QLoRA 4-bit NormalFloat (NF4)
* **Target Linear Projections:** Attention query and value projections (`q_proj`, `v_proj`)
* **LoRA Hyperparameters:**
  * Rank ($r$): $16$
  * Alpha ($\alpha$): $32$
  * Dropout: $0.05$
* **Precision Policy:** Mixed-Precision FP16 compute
* **Acoustic Preprocessing:** Downsampled to 16,000 Hz single-channel PCM; sequence token targets truncated to 448 decoder tokens.