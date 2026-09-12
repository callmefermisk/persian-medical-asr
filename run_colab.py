"""
Google Colab Runner for Persian Medical ASR
Launches the Gradio interface with a public link and verifies GPU acceleration.
"""

import torch
from app import demo

if __name__ == "__main__":
    print("=" * 55)
    print("Persian Medical ASR - Google Colab Engine")
    print("=" * 55)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[+] CUDA is available. Running on: {gpu_name}")
    else:
        print("[!] Warning: Running on CPU. For faster inference, switch runtime:")
        print("    Runtime -> Change runtime type -> T4 GPU")

    print("\n[+] Launching interactive web demo...")
    # share=True creates a public gradio.live link accessible outside Colab
    demo.launch(share=True, debug=True)