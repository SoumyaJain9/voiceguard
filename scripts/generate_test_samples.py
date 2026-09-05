import os
import wave
import struct
import math
import sys
from pathlib import Path

# Safe UTF-8 console output for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def generate_sample_audios(output_dir: str = "sample_audio"):
    """
    Generates synthetic tones and organic simulated vocal signals 
    using Python standard library (wave & struct) without external dependencies.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    sr = 16000
    duration = 3.0
    num_samples = int(sr * duration)

    # 1. Synthetic / Monotone Robotic Tone (Pure constant fundamental frequency + harmonics)
    synth_file = out_path / "sample_synthetic_robotic.wav"
    with wave.open(str(synth_file), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sr)

        f0 = 220.0  # Constant A3 pitch
        for i in range(num_samples):
            t = float(i) / sr
            sample_val = math.sin(2 * math.pi * f0 * t) + 0.3 * math.sin(2 * math.pi * 2 * f0 * t)
            int_val = int(sample_val * 16000.0)
            int_val = max(-32767, min(32767, int_val))
            wav_file.writeframes(struct.pack("<h", int_val))

    print(f"[+] Generated synthetic demonstration audio: {synth_file}")

    # 2. Organic-style modulated signal (Emulated glottal perturbation and breath amplitude variation)
    organic_file = out_path / "sample_organic_human.wav"
    with wave.open(str(organic_file), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)

        current_phase = 0.0
        for i in range(num_samples):
            t = float(i) / sr
            f_inst = 180.0 + 4.0 * math.sin(2 * math.pi * 5.5 * t)
            current_phase += 2 * math.pi * f_inst / sr
            envelope = 0.8 + 0.2 * math.sin(2 * math.pi * 1.5 * t)
            sample_val = envelope * math.sin(current_phase)
            
            int_val = int(sample_val * 16000.0)
            int_val = max(-32767, min(32767, int_val))
            wav_file.writeframes(struct.pack("<h", int_val))

    print(f"[+] Generated organic demonstration audio:   {organic_file}")

if __name__ == "__main__":
    generate_sample_audios()
