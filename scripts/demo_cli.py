import os
import sys
import argparse
import base64
import requests
import json
from pathlib import Path

# Safe UTF-8 console output for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.inference import processor, InvalidAudioError
from src.api.config import settings

def run_local_analysis(audio_path: str, language: str = "English"):
    """Runs direct local inference and forensic feature extraction on an audio file."""
    print("=" * 70)
    print("VOXGUARD AI - DEEPFAKE AUDIO FORENSIC INSPECTOR (LOCAL ENGINE)")
    print("=" * 70)
    print(f"Target Audio: {audio_path}")
    print(f"Language:     {language}")
    print("-" * 70)

    if not os.path.exists(audio_path):
        print(f"[-] Error: Audio file '{audio_path}' not found.")
        return

    try:
        waveform = processor.preprocess(audio_path)
        confidence = processor.predict(waveform)
        
        label = "AI_GENERATED" if confidence > 0.50 else "HUMAN"
        display_confidence = confidence if label == "AI_GENERATED" else (1.0 - confidence)

        analysis = processor.analyze_features(audio_path, prediction_label=label, prediction_score=confidence)

        # Print Formatted Results
        print(f"\n[*] VERDICT:       {'[AI GENERATED / SPOOFED]' if label == 'AI_GENERATED' else '[GENUINE HUMAN SPEECH]'}")
        print(f"[*] CONFIDENCE:    {display_confidence * 100:.2f}%")
        print(f"\n[*] ACOUSTIC FORENSIC BIOMARKERS:")
        print(f"   |-- Jitter (Pitch Stability):        {analysis['jitter'] * 100:.3f}% (Normal Range: 0.20% - 1.50%)")
        print(f"   |-- Shimmer (Amplitude Dynamics):    {analysis['shimmer'] * 100:.3f}% (Normal Threshold: < 5.0%)")
        print(f"   |-- Harmonics-to-Noise Ratio (HNR):  {analysis['hnr']:.2f} dB (Clean Speech > 20 dB)")
        print(f"   \\-- Spectral Flatness:              {analysis['spectral_flatness']:.5f}")
        
        print(f"\n[*] DECISION WEIGHTS:")
        print(f"   |-- Deep Neural Representation:      {analysis['confidence_weights']['Neural_Pattern_Match'] * 100:.1f}%")
        print(f"   \\-- Digital Signal Artifacts:        {analysis['confidence_weights']['Acoustic_Signal_Artifacts'] * 100:.1f}%")

        print(f"\n[*] FORENSIC EXPLANATION:")
        print(f"   {analysis['text']}")
        print("=" * 70)

    except Exception as e:
        print(f"[-] Analysis failed: {e}")

def run_api_analysis(audio_path: str, api_url: str = "http://localhost:8000/api/voice-detection", language: str = "English", api_key: str = "voxguard-college-eval-key"):
    """Tests the running FastAPI server endpoint via HTTP."""
    print("=" * 70)
    print(f"VOXGUARD AI - TESTING REST API AT: {api_url}")
    print("=" * 70)

    if not os.path.exists(audio_path):
        print(f"[-] Error: File '{audio_path}' not found.")
        return

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "language": language,
        "audioFormat": audio_path.split(".")[-1],
        "audioBase64": audio_b64
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }

    try:
        res = requests.post(api_url, json=payload, headers=headers, timeout=15)
        print(f"Status Code: {res.status_code}")
        print("Response JSON:")
        print(json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"[-] Request to API failed: {e}. (Ensure server is running with 'uvicorn src.api.main:app')")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxGuard AI Terminal Demonstration Tool")
    parser.add_argument("--file", help="Path to audio file (.wav, .mp3, .flac)")
    parser.add_argument("--language", default="English", help="Audio language")
    parser.add_argument("--api", action="store_true", help="Send request to running HTTP API instead of local analysis")
    parser.add_argument("--url", default="http://localhost:8000/api/voice-detection", help="API URL")
    args = parser.parse_args()

    if not args.file:
        sample_dir = Path(__file__).resolve().parent.parent / "sample_audio"
        test_file = sample_dir / "sample_synthetic_robotic.wav"
        if not test_file.exists():
            from generate_test_samples import generate_sample_audios
            generate_sample_audios(str(sample_dir))
        args.file = str(test_file)

    if args.api:
        run_api_analysis(args.file, api_url=args.url, language=args.language)
    else:
        run_local_analysis(args.file, language=args.language)
