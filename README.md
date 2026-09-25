# Persian Medical Speech Recognition (ASR) & Clinical Speaker Diarization
### Domain-Adapted Whisper Large-v3 via LoRA & Speaker-Attributed Dialogue Transcription

This repository provides an automated speech-to-text and speaker diarization pipeline tailored for Persian clinical consultations, medical dictations, and patient-doctor interactions. The system builds upon `nezamisafa/whisper-persian-v4` (Whisper Large-v3), integrating a domain-adapted LoRA adapter alongside neural speaker diarization (`pyannote.audio 3.1`).

---

## Benchmark & Performance Evaluation

The ASR model was evaluated on a held-out test split of clinical consultation recordings (113 samples). Text representations were normalized using `hazm` and stripped of confounding punctuation before metric calculation.

| Metric | Baseline (`whisper-persian-v4`) | Medical LoRA (`v2 Adapter`) | Relative Gain |
| :--- | :---: | :---: | :---: |
| **Word Error Rate (WER)** | 50.11% | **42.12%** | **~16.0% Relative Reduction** |
| **Semantic Cosine Similarity** | 75.89% | **85.56%** | **+9.67% Absolute Increase** |

* **Semantic Fidelity:** Semantic similarity scores were measured using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The jump to **85.56%** indicates superior retention of clinical intent, medication names, and diagnostic descriptions.
* **Error Rate:** The 8% absolute WER drop demonstrates the adapter's capacity to recognize specialized medical terminology and informal Persian conversational syntax without catastrophic forgetting.

---

## Repository Structure

```text
persian-medical-asr/
│
├── medical_adapter/               # Fine-tuned LoRA weights (~30MB)
│   ├── adapter_config.json        # LoRA rank and target module definitions
│   ├── adapter_model.safetensors  # Trainable delta weights
│   ├── processor_config.json      # Audio feature extraction parameters
│   ├── tokenizer_config.json      # Whisper BPE tokenizer configuration
│   └── tokenizer.json             # Vocabulary mappings
│
├── sample.wav                     # Single-speaker verification sample
├── generate_sample.py             # Dual-speaker synthetic clinical dialogue generator
├── diarized_transcribe.py         # Multi-speaker diarization + ASR engine (CLI)
├── transcribe.py                  # Single-speaker inference engine (CLI)
├── app.py                         # Interactive Gradio web interface
├── run_colab.py                   # Google Colab entrypoint with public Gradio link
├── requirements.txt               # Package dependencies
└── README.md                      # Project documentation and reproduction steps

```

---

## Quick Start: Google Colab Execution

The pipeline is pre-configured for instant zero-setup evaluation on Google Colab with a free T4 GPU.

### Step 1: Clone and Install

```bash
!rm -rf persian-medical-asr
!git clone [https://github.com/callmefermisk/persian-medical-asr.git](https://github.com/callmefermisk/persian-medical-asr.git)
%cd persian-medical-asr
!pip install -q -r requirements.txt

```

### Step 2: Generate a Dual-Speaker Consultation Sample

Synthesizes a realistic Persian clinical encounter (male physician, female patient) via neural TTS:

```bash
!python generate_sample.py

```

### Step 3: Run Inference

* **Multi-Speaker Diarized Transcription (Doctor & Patient):**
```bash
!python diarized_transcribe.py --audio sample_consultation.wav

```


*(Pre-configured authentication handles gated pipeline weights automatically without manual token entry).*
* **Single-Speaker Clinical Dictation:**
```bash
!python transcribe.py

```


* **Interactive Web App (Public Gradio Link):**
```bash
!python run_colab.py

```



---

## Local Environment Setup

### 1. Prerequisites

* Python 3.9 or higher
* Recommended: NVIDIA GPU with CUDA support (CPU execution is fully supported)
* `ffmpeg` installed on your system PATH:
* **Ubuntu/Debian:** `sudo apt-get install ffmpeg`
* **macOS:** `brew install ffmpeg`
* **Windows:** `winget install Gyan.FFmpeg`



### 2. Installation

```bash
git clone [https://github.com/callmefermisk/persian-medical-asr.git](https://github.com/callmefermisk/persian-medical-asr.git)
cd persian-medical-asr
pip install -r requirements.txt

```

---

## Local Usage & CLI Execution

### Mode A: Multi-Speaker Clinical Diarization (Doctor vs. Patient)

Processes multi-speaker audio, detects who spoke when, and aligns transcripts with speaker identities:

```bash
# Transcribe the bundled synthetic sample
python diarized_transcribe.py --audio sample_consultation.wav

# Transcribe any consultation recording
python diarized_transcribe.py \
    --audio path/to/consultation.wav \
    --num_speakers 2 \
    --output dialogue_transcript.txt

```

#### CLI Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--audio` | `str` | *Required* | Path to input audio (`.wav`, `.mp3`, `.m4a`). |
| `--num_speakers` | `int` | `2` | Expected number of distinct speakers. |
| `--hf_token` | `str` | *Auto* | Optional custom Hugging Face token for PyAnnote. |
| `--model_id` | `str` | `nezamisafa/whisper-persian-v4` | Base Hugging Face model repository. |
| `--output` | `str` | `medical_transcript.txt` | File path to export transcript output. |

### Mode B: Single-Speaker Clinical Dictation

For single-speaker recordings, doctor notes, or medical dictations:

```bash
# Verify with default sample
python transcribe.py

# Transcribe custom audio
python transcribe.py path/to/doctor_memo.wav

```

### Mode C: Interactive Web Application

Launches a browser interface for microphone recording and audio file analysis:

```bash
python app.py

```

*Access the interface locally at `http://127.0.0.1:7860`.*

---

## Multi-Speaker Architecture

```text
[Consultation Audio]
         │
         ├──► [PyAnnote Diarization 3.1] ──► Speaker timestamps & turn boundaries
         │                                              │
         └──► [Chunking & Resampling (16 kHz)]          │
                     │                                  │
                     ▼                                  ▼
         [Whisper Medical LoRA Engine] ◄──────── [Dialogue Merging & Alignment]
                     │
                     ▼
         [Speaker-Attributed Clinical Transcript]

```

* **Biometric Acoustic Clustering:** Isolates vocal tract characteristics to track speaker identities across voice modulations and natural turn-taking.
* **Contextual Turn Aggregation:** Merges fragmented sub-utterances by the same speaker separated by minor pauses ($< 0.8\text{ s}$) to prevent context loss in the Whisper decoder.
* **Clinical Lexical Steering:** Injects domain-specific token prompt biasing during generation to preserve Persian pharmaceutical names (e.g., ژلوفن, استامینوفن, آسپرین).

---

## Sample Transcript Output

```text
==================================================
نتایج بازشناسی مکالمه بالینی:
==================================================
[00:00.50s -> 00:04.10s] SPEAKER_00: سلام، روزتون بخیر. بفرمایید بنشینید، مشکلتون از چه زمانی شروع شده؟
[00:04.80s -> 00:09.60s] SPEAKER_01: سلام آقای دکتر. از دیروز سوزش و درد شدید معده دارم و هر چی می‌خورم حالم بدتر می‌شه.
[00:10.20s -> 00:13.40s] SPEAKER_00: آیا سابقه مصرف داروی خاصی مثل آسپرین یا بروفن رو در این چند روز داشتید؟
[00:14.10s -> 00:16.80s] SPEAKER_01: بله، برای دندون‌دردم چند تا ژلوفن مصرف کردم.

```

---

## Training Methodology

* **Base Architecture:** `nezamisafa/whisper-persian-v4` (Whisper Large-v3)
* **Optimization Framework:** Parameter-Efficient Fine-Tuning (PEFT / QLoRA 4-bit)
* **Target Modules:** Query and Value projection layers (`q_proj`, `v_proj`)
* **LoRA Parameters:** $r = 16$, $\alpha = 32$, Dropout = $0.05$
* **Precision:** Mixed Precision FP16
* **Audio Preprocessing:** 16,000 Hz single-channel PCM waveforms; labels truncated to 448 decoder tokens.

```

```
