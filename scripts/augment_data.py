import os
import argparse
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm

def add_gaussian_noise(audio: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    """Injects calibrated additive white Gaussian noise (AWGN) to simulate transmission line noise."""
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    augmented = audio + noise
    return augmented.astype(np.float32)

def augment_dataset(root_dir: str = "Dataset"):
    """
    Applies speed perturbations and noise injection across all speech categories.
    Simulates real-world telephony channels, VoIP compression, and microphone artifacts.
    """
    categories = ['real', 'fake']
    languages = ['en', 'hi', 'ta', 'te', 'ml']

    print(f"🛠️ Starting VoxGuard Acoustic Augmentation in: '{root_dir}'...")

    for category in categories:
        for lang in languages:
            target_dir = os.path.join(root_dir, category, lang)
            if not os.path.exists(target_dir):
                continue

            print(f"Processing partition: {category}/{lang}...")
            files = [f for f in os.listdir(target_dir) if f.endswith('.wav') and '_aug' not in f]

            for filename in tqdm(files, desc=f"Augmenting {category}/{lang}", leave=False):
                file_path = os.path.join(target_dir, filename)
                try:
                    y, sr = sf.read(file_path)

                    # 1. Speed Perturbation (Resampling -> 0.9x / 1.1x speed variations)
                    speed_name = filename.replace(".wav", "_aug_speed.wav")
                    speed_path = os.path.join(target_dir, speed_name)
                    if not os.path.exists(speed_path):
                        speed = float(np.random.choice([0.9, 1.1]))
                        y_speed = librosa.resample(y=y, orig_sr=sr, target_sr=int(sr * speed))
                        sf.write(speed_path, y_speed, sr)

                    # 2. Additive White Gaussian Noise
                    noise_name = filename.replace(".wav", "_aug_noise.wav")
                    noise_path = os.path.join(target_dir, noise_name)
                    if not os.path.exists(noise_path):
                        y_noise = add_gaussian_noise(y, snr_db=float(np.random.uniform(10.0, 20.0)))
                        sf.write(noise_path, y_noise, sr)

                except Exception as e:
                    continue

    print("✅ Acoustic Augmentation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxGuard Acoustic Data Augmentation Engine")
    parser.add_argument("--output_dir", default="Dataset", help="Root folder of the dataset")
    args = parser.parse_args()

    augment_dataset(args.output_dir)
