import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ModelWithTemperature(nn.Module):
    """
    Temperature Scaling Calibration Wrapper.
    Implements post-processing calibration to ensure the output probabilities 
    accurately reflect true empirical correctness likelihood, mitigating overconfidence.
    """
    def __init__(self, model: nn.Module):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        # Initialize temperature parameter to 1.5
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_tensor)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies temperature division to raw classification logits."""
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader: DataLoader, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Tunes the scalar temperature parameter on the validation dataset using L-BFGS.
        Minimizes Negative Log Likelihood (NLL) and Expected Calibration Error (ECE).
        """
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        # 1. Accumulate validation logits and target labels
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels.to(device))

        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        # 2. Evaluate uncalibrated baseline metrics
        before_nll = nll_criterion(logits, labels).item()
        before_ece = ece_criterion(logits, labels).item()
        print(f"📊 Before Calibration -> NLL: {before_nll:.4f} | ECE: {before_ece:.4f}")

        # 3. Optimize temperature using L-BFGS optimizer
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_step():
            optimizer.zero_grad()
            scaled_logits = self.temperature_scale(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        # 4. Evaluate calibrated results
        scaled_logits = self.temperature_scale(logits)
        after_nll = nll_criterion(scaled_logits, labels).item()
        after_ece = ece_criterion(scaled_logits, labels).item()
        print(f"🎯 Optimal Temperature Parameter T = {self.temperature.item():.4f}")
        print(f"✅ After Calibration  -> NLL: {after_nll:.4f} | ECE: {after_ece:.4f}")

        return self

class _ECELoss(nn.Module):
    """
    Expected Calibration Error (ECE) Evaluation Metric.
    Divides confidence scores into discrete bins and measures absolute deviation
    between bin average confidence and empirical accuracy.
    """
    def __init__(self, n_bins: int = 15):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        softmaxes = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
