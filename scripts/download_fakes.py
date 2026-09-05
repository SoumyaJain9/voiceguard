import os
import argparse
import soundfile as sf
import librosa
import numpy as np
import io
from datasets import load_dataset, Audio
from tqdm import tqdm

def process_audio_bytes(audio_bytes: bytes, target_sr: int = 16000):
    """Converts audio bytes into standardized mono 16kHz float32 arrays."""
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        return None, None

    if y.dtype != np.float32:
        y = y.astype(np.float32)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1) if y.shape[1] > 1 else y.flatten()

    if sr != target_sr:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y, sr

def download_synthetic_corpus(output_dir: str = "Dataset", limit_per_lang: int = 500):
    """
    Downloads and prepares synthetic AI-generated audio spoofing attempts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. English Synthetic Audio
    print("\n🤖 [1/2] Processing English Synthetic Speech...")
    en_fake_dir = os.path.join(output_dir, "fake", "en")
    os.makedirs(en_fake_dir, exist_ok=True)

    existing_en = len([f for f in os.listdir(en_fake_dir) if f.endswith('.wav')])
    if existing_en < limit_per_lang:
        try:
            ds = load_dataset("garystafford/deepfake-audio-detection", split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))

            count = existing_en
            pbar = tqdm(total=limit_per_lang, initial=count, desc="English Synthetic")

            for sample in ds:
                if count >= limit_per_lang:
                    break

                label = sample.get('label')
                if label != 1 and label != 'fake':
                    continue

                if 'audio' in sample:
                    y, sr = process_audio_bytes(sample['audio']['bytes'])
                    if y is None:
                        continue

                    out_path = os.path.join(en_fake_dir, f"fake_en_{count:04d}.wav")
                    if not os.path.exists(out_path):
                        sf.write(out_path, y, sr)
                    count += 1
                    pbar.update(1)
            pbar.close()
        except Exception as e:
            print(f"⚠️ English synthetic stream error: {e}")
    else:
        print(f"English synthetic partition already has {existing_en} samples.")

    # 2. Indic Synthetic Audio (IndicSynth)
    print("\n🇮🇳 [2/2] Processing Indic Synthetic Audio (IndicSynth)...")
    indic_langs = {
        "hi": "Hindi",
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam"
    }

    for code, name in indic_langs.items():
        lang_dir = os.path.join(output_dir, "fake", code)
        os.makedirs(lang_dir, exist_ok=True)

        existing = len([f for f in os.listdir(lang_dir) if f.endswith('.wav')])
        if existing >= limit_per_lang:
            print(f"Synthetic {name} ({code}) partition complete ({existing} samples).")
            continue

        try:
            print(f"Streaming {name} ({code}) from IndicSynth...")
            ds = load_dataset("vdivyasharma/IndicSynth", code, split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))

            count = existing
            pbar = tqdm(total=limit_per_lang, initial=count, desc=f"Synthetic {name}")

            for sample in ds:
                if count >= limit_per_lang:
                    break

                if 'audio' in sample:
                    y, sr = process_audio_bytes(sample['audio']['bytes'])
                    if y is None:
                        continue

                    out_path = os.path.join(lang_dir, f"fake_{code}_{count:04d}.wav")
                    if not os.path.exists(out_path):
                        sf.write(out_path, y, sr)
                    count += 1
                    pbar.update(1)
            pbar.close()
        except Exception as e:
            print(f"⚠️ IndicSynth stream notice for {name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxGuard Synthetic Speech Harvester")
    parser.add_argument("--output_dir", default="Dataset", help="Directory to save synthetic audio")
    parser.add_argument("--limit", type=int, default=500, help="Max files per language")
    args = parser.parse_args()

    download_synthetic_corpus(output_dir=args.output_dir, limit_per_lang=args.limit)
