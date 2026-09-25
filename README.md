# Persian Medical Speech Recognition (ASR)
### Domain Adaptation of Whisper Large-v3 via Low-Rank Adaptation (LoRA)

This repository provides an automated speech-to-text pipeline tailored for Persian clinical consultations, patient inquiries, and medical terminology. The system builds upon `nezamisafa/whisper-persian-v4` (Whisper Large-v3) and integrates a domain-adapted LoRA adapter trained on conversational medical interactions.

---

## Benchmark & Performance Evaluation

The model was evaluated on a held-out test split of clinical dialogue recordings (113 samples). Text representations were normalized using `hazm` and stripped of confounding punctuation before metric calculation.

| Metric | Baseline (`whisper-persian-v4`) | Medical LoRA (`v2 Adapter`) | Relative Gain |
| :--- | :---: | :---: | :---: |
| **Word Error Rate (WER)** | 50.11% | **42.12%** | **~16.0% Relative Reduction** |
| **Semantic Cosine Similarity** | 75.89% | **85.56%** | **+9.67% Absolute Increase** |

* **Semantic Fidelity:** Semantic similarity scores were measured using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The jump to **85.56%** indicates superior retention of clinical intent, medication names, and diagnosis descriptions.
* **Error Rate:** The 8% absolute WER drop demonstrates the adapter's capacity to recognize medical entities and informal conversational syntax without catastrophic forgetting.

---

## Repository Structure

```text
persian_medical_asr/
│
├── medical_adapter/               # Fine-tuned LoRA weights (~30MB)
│   ├── adapter_config.json        # LoRA rank and target module definitions
│   ├── adapter_model.safetensors  # Trainable delta weights
│   ├── processor_config.json      # Audio feature extraction parameters
│   ├── tokenizer_config.json      # Whisper BPE tokenizer configuration
│   └── tokenizer.json             # Vocabulary mappings
│
├── sample.wav                     # Verification audio sample (Persian consultation)
├── transcribe.py                  # Single-speaker inference engine (CLI)
├── diarized_transcribe.py         # Multi-speaker diarization + ASR engine (CLI)
├── generate_sample.py             # Synthetic 2-speaker consultation audio generator
├── app.py                         # Interactive Gradio web interface
├── run_colab.py                   # Google Colab entrypoint with public link
├── requirements.txt               # Package dependencies
└── README.md                      # Project documentation and reproduction steps

```

---

## Installation & Setup (Local Environment)

### 1. Prerequisites

* Python 3.9 or higher
* Recommended: NVIDIA GPU with CUDA support (CPU execution is fully supported out of the box).
* `ffmpeg` installed on your system PATH (required for audio resampling and normalization).

### 2. Install Dependencies

Clone the repository and install the required packages:

```bash
git clone [https://github.com/callmefermisk/persian-medical-asr.git](https://github.com/callmefermisk/persian-medical-asr.git)
cd persian-medical-asr
pip install -r requirements.txt

```

---

## Local Usage & Inference

The inference scripts automatically detect hardware accelerators (CUDA/CPU), format incoming waveforms to 16 kHz mono, dynamically mount the LoRA adapter, and output decoded text.

### Option 1: Interactive Web UI (Microphone & Upload)

Launch the local web application:

```bash
python app.py

```

*Opens an interactive interface at `http://127.0.0.1:7860` for live microphone recording and audio file transcription.*

### Option 2: Single-Speaker Verification Test

Transcribe the bundled clinical sample:

```bash
python transcribe.py

```

### Option 3: Transcribe Any Single-Speaker File

Pass the path of any `.wav` or `.mp3` recording:

```bash
python transcribe.py path/to/consultation.wav

```

> **Note on Initial Run:** During the first execution, Hugging Face will automatically download and cache the base Whisper Large-v3 model (`nezamisafa/whisper-persian-v4`, ~3 GB). All subsequent runs execute offline.

---

## Running Single-Speaker ASR on Google Colab

You can run this project on Google Colab with a free T4 GPU without any local environment setup.

### Step 1: Enable Hardware Accelerator

1. Open a new notebook at [Google Colab](https://colab.research.google.com?utm_source=gemini).
2. Go to **Runtime** > **Change runtime type**.
3. Select **T4 GPU** under Hardware accelerator and click **Save**.

### Step 2: Clone and Install

Run this in the first Colab cell:

```bash
!git clone [https://github.com/callmefermisk/persian-medical-asr.git](https://github.com/callmefermisk/persian-medical-asr.git)
%cd persian-medical-asr
!pip install -q -r requirements.txt

```

### Step 3: Run Inference

* **Option A: Interactive Web Demo (Public Gradio Link)**

```bash
!python run_colab.py

```

*Click the generated public `https://xxxx.gradio.live` URL to test via microphone or file upload directly from your browser.*

* **Option B: Quick CLI Verification**

```bash
!python transcribe.py

```

---

## Training Methodology

* **Base Model:** `nezamisafa/whisper-persian-v4` (Whisper Large-v3)
* **Optimization Method:** Parameter-Efficient Fine-Tuning (PEFT / QLoRA 4-bit)
* **Target Modules:** Query and Value projection layers (`q_proj`, `v_proj`)
* **LoRA Parameters:** $r = 16$, $\alpha = 32$, Dropout = $0.05$
* **Precision:** Mixed Precision FP16
* **Preprocessing:** Waveforms downsampled to 16,000 Hz single-channel; labels truncated to 448 decoder tokens.

---

## Multi-Speaker Clinical Diarization (Doctor-Patient Attribution)

For real-world clinical consultations involving multiple speakers, the system provides a **Speaker-Attributed ASR** pipeline (`diarized_transcribe.py`). This pipeline decouples **acoustic speaker segmentation** (identifying *who spoke when*) from **domain-adapted transcription** (decoding *what was spoken*).

```text
[Multi-Speaker Audio]
         │
         ├──► [PyAnnote Diarization 3.1] ──► Speaker timestamps & turn segmentation
         │                                               │
         └──► [Chunking & Resampling (16kHz)]            │
                     │                                   │
                     ▼                                   ▼
         [Whisper Medical LoRA Engine] ◄──────── [Dialogue Merging & Alignment]
                     │
                     ▼
         [Speaker-Attributed Clinical Transcript]

```

### Pipeline Features

* **Biometric Acoustic Clustering:** Leverages neural speaker embeddings to reliably differentiate dialogue turns between physician and patient even across voice modulations and natural turn-taking.
* **Turn Aggregation:** Automatically merges fragmented utterances by the same speaker separated by minor pauses ($< 0.8\text{ s}$) to preserve contextual coherence for the Whisper decoder.
* **Clinical Lexical Steering:** Employs domain-specific token prompt biasing during decoding to maintain accurate spelling of Persian pharmaceutical terms (e.g., ژلوفن, استامینوفن, آسپرین).

---

### Prerequisites & Access Permissions

The diarization engine utilizes `pyannote/speaker-diarization-3.1`. Due to gating restrictions on Hugging Face, accept the user conditions once on both models using your Hugging Face account:

1. Accept conditions for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1?utm_source=gemini).
2. Accept conditions for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0?utm_source=gemini).
3. Generate a Hugging Face user access token (`Read` permission) under **Settings** > **Access Tokens**.

---

### Quick Start: Multi-Speaker Diarization on Google Colab

Run the entire multi-speaker consultation pipeline inside Google Colab in three steps:

#### 1. Setup Environment

```bash
!git clone [https://github.com/callmefermisk/persian-medical-asr.git](https://github.com/callmefermisk/persian-medical-asr.git)
%cd persian-medical-asr
!pip install -q -r requirements.txt

```

#### 2. Generate Multi-Speaker Audio Sample

If you do not have a recorded consultation audio file readily available, generate a dual-speaker clinical scenario (male physician, female patient) via neural TTS:

```bash
!python generate_sample.py

```

*Outputs `sample_consultation.wav` (16 kHz mono WAV) directly into the workspace.*

#### 3. Execute Diarized Transcription

```bash
!python diarized_transcribe.py \
    --audio sample_consultation.wav \
    --hf_token "YOUR_HF_TOKEN" \
    --num_speakers 2 \
    --output consultation_transcript.txt

```

---

### Local Execution: Multi-Speaker Diarization

Execute the pipeline from your local terminal:

```bash
# Transcribe multi-speaker consultation
python diarized_transcribe.py \
    --audio path/to/consultation.wav \
    --hf_token "YOUR_HF_TOKEN" \
    --num_speakers 2 \
    --output dialogue_output.txt

```

#### CLI Parameters

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--audio` | `str` | *Required* | Path to the target audio file (`.wav`, `.mp3`, `.m4a`). |
| `--hf_token` | `str` | *Required* | Hugging Face Access Token with read access to PyAnnote. |
| `--num_speakers` | `int` | `2` | Expected number of distinct speakers (set `2` for doctor-patient). |
| `--model_id` | `str` | `nezamisafa/whisper-persian-v4` | Base Hugging Face model identifier for Whisper. |
| `--output` | `str` | `medical_transcript.txt` | Output destination for the formatted text transcript. |

---

### Sample Output

```text
==================================================
Clinical Dialogue Transcript (Speaker-Attributed ASR)
==================================================

[00:00.50s -> 00:04.10s] SPEAKER_00:
  سلام، روزتون بخیر. بفرمایید بنشینید، مشکلتون از چه زمانی شروع شده؟

[00:04.80s -> 00:09.60s] SPEAKER_01:
  سلام آقای دکتر. از دیروز سوزش و درد شدید معده دارم و هر چی می‌خورم حالم بدتر می‌شه.

[00:10.20s -> 00:13.40s] SPEAKER_00:
  آیا سابقه مصرف داروی خاصی مثل آسپرین یا بروفن رو در این چند روز داشتید؟

[00:14.10s -> 00:16.80s] SPEAKER_01:
  بله، برای دندون‌دردم چند تا ژلوفن مصرف کردم.

```

```

```
