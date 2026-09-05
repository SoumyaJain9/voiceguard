import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List

class DeepfakeDataset(Dataset):
    """
    Multilingual Audio Dataset Loader for Deepfake Detection.
    Loads real (genuine human) and fake (AI synthesized) speech samples,
    standardizes sampling rates to 16kHz mono, and applies random 3-second windowing.
    """
    def __init__(self, data_dir: str, split: str = "train", max_samples: Optional[int] = None, target_sr: int = 16000):
        """
        Args:
            data_dir: Path to dataset directory containing 'real' and 'fake' subfolders.
            split: Dataset split ('train', 'dev', or 'eval').
            max_samples: Maximum samples to load for quick debugging/smoke tests.
            target_sr: Standardized audio sample rate (default: 16000 Hz).
        """
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.files: List[str] = []
        self.labels: List[int] = []

        # Real samples (Label = 0), Fake samples (Label = 1)
        real_pattern = os.path.join(data_dir, "real", "**", "*.wav")
        fake_pattern = os.path.join(data_dir, "fake", "**", "*.wav")

        real_files = glob.glob(real_pattern, recursive=True)
        fake_files = glob.glob(fake_pattern, recursive=True)

        if max_samples:
            real_files = real_files[:max_samples]
            fake_files = fake_files[:max_samples]

        print(f"📊 [Dataset] Found {len(real_files)} Genuine Human speech samples")
        print(f"📊 [Dataset] Found {len(fake_files)} Synthetic AI Voice samples")

        self.files = real_files + fake_files
        self.labels = [0] * len(real_files) + [1] * len(fake_files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(file_path)

            # Resample to model sample rate (16kHz)
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)

            # Stereo to Mono conversion
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Fixed length windowing: 3 seconds = 48,000 samples
            max_len = 48000
            if waveform.shape[1] > max_len:
                # Random window cropping for training variance
                start = torch.randint(0, waveform.shape[1] - max_len, (1,)).item()
                waveform = waveform[:, start:start + max_len]
            elif waveform.shape[1] < max_len:
                # Zero padding for shorter audio clips
                padding = max_len - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            return waveform.squeeze(0), torch.tensor(label, dtype=torch.long)

        except Exception as e:
            # Fallback zero tensor to prevent training crashes on corrupted files
            print(f"⚠️ Error reading {file_path}: {e}")
            return torch.zeros(48000), torch.tensor(label, dtype=torch.long)

def get_dataloaders(data_dir: str, batch_size: int = 32, val_split: float = 0.1, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoader instances for training and validation splits.
    """
    full_dataset = DeepfakeDataset(data_dir)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
