import os
import argparse
import soundfile as sf
import librosa
import numpy as np
import io
from datasets import load_dataset, Audio
from tqdm import tqdm

def process_audio_bytes(audio_bytes: bytes, target_sr: int = 16000):
    """Decodes raw audio bytes and converts to single-channel 16kHz float32 numpy array."""
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        return None, None

    if y.dtype != np.float32:
        y = y.astype(np.float32)

    # Convert Stereo to Mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1) if y.shape[1] > 1 else y.flatten()

    # Resample to 16kHz
    if sr != target_sr:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y, sr

def download_real_corpus(output_dir: str = "Dataset", limit_per_lang: int = 500, common_voice_archive_dir: str = "data/common_voice"):
    """
    Downloads and standardizes genuine human speech recordings for English and Indic languages.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. English Human Speech Baseline
    print("\n🎙️ [1/2] Processing Genuine Human Speech (English)...")
    en_dir = os.path.join(output_dir, "real", "en")
    os.makedirs(en_dir, exist_ok=True)

    existing_en = len([f for f in os.listdir(en_dir) if f.endswith('.wav')])
    if existing_en < limit_per_lang:
        try:
            ds = load_dataset("garystafford/deepfake-audio-detection", split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))

            count = existing_en
            pbar = tqdm(total=limit_per_lang, initial=count, desc="English Real")

            for sample in ds:
                if count >= limit_per_lang:
                    break

                label = sample.get('label', 0)
                if label == 1 or label == 'fake':
                    continue

                if 'audio' in sample:
                    y, sr = process_audio_bytes(sample['audio']['bytes'])
                    if y is None:
                        continue

                    out_path = os.path.join(en_dir, f"real_en_{count:04d}.wav")
                    if not os.path.exists(out_path):
                        sf.write(out_path, y, sr)
                    count += 1
                    pbar.update(1)
            pbar.close()
        except Exception as e:
            print(f"⚠️ English stream error: {e}")
    else:
        print(f"English partition already contains {existing_en} files. Skipping.")

    # 2. Indic Common Voice Corpora
    print("\n🇮🇳 [2/2] Checking Indic Language Corpora (Hindi, Tamil, Telugu, Malayalam)...")
    indic_langs = {
        'hi': 'mcv-scripted-hi-v24.0.tar.gz',
        'ta': 'mcv-scripted-ta-v24.0.tar.gz',
        'te': 'mcv-scripted-te-v24.0.tar.gz',
        'ml': 'mcv-scripted-ml-v24.0.tar.gz'
    }

    for lang_code, tar_name in indic_langs.items():
        lang_dir = os.path.join(output_dir, "real", lang_code)
        os.makedirs(lang_dir, exist_ok=True)
        existing = len([f for f in os.listdir(lang_dir) if f.endswith('.wav')])
        if existing >= limit_per_lang:
            print(f"Language '{lang_code}' already contains {existing} files. Skipping.")
            continue

        tar_path = os.path.join(common_voice_archive_dir, tar_name)
        if os.path.exists(tar_path):
            import tarfile
            print(f"Extracting {lang_code} from local archive: {tar_name}")
            with tarfile.open(tar_path, "r:gz") as tar:
                count = existing
                for member in tar:
                    if count >= limit_per_lang:
                        break
                    if member.name.endswith(".mp3"):
                        f = tar.extractfile(member)
                        if f:
                            y, sr = process_audio_bytes(f.read())
                            if y is not None:
                                sf.write(os.path.join(lang_dir, f"real_{lang_code}_{count:04d}.wav"), y, sr)
                                count += 1
            print(f"Extracted {count} samples for {lang_code}.")
        else:
            print(f"ℹ️ Optional archive '{tar_name}' not found locally in '{common_voice_archive_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxGuard Real Human Voice Corpus Harvester")
    parser.add_argument("--output_dir", default="Dataset", help="Directory to save extracted audio")
    parser.add_argument("--limit", type=int, default=500, help="Max files per language")
    args = parser.parse_args()

    download_real_corpus(output_dir=args.output_dir, limit_per_lang=args.limit)
