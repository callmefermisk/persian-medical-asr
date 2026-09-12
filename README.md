
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
├── transcribe.py                  # Standalone inference engine (CLI)
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

### Option 2: CLI Verification Test

Transcribe the bundled clinical sample:

```bash
python transcribe.py

```

### Option 3: Transcribe Any Audio File

Pass the path of any `.wav` or `.mp3` recording:

```bash
python transcribe.py path/to/consultation.wav

```

> **Note on Initial Run:** During the first execution, Hugging Face will automatically download and cache the base Whisper Large-v3 model (`nezamisafa/whisper-persian-v4`, ~3 GB). All subsequent runs execute offline.

---

## Running on Google Colab

You can run this project on Google Colab with a free T4 GPU without any local environment setup.

### Step 1: Enable Hardware Accelerator

1. Open a new notebook at [Google Colab](https://colab.research.google.com).
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
