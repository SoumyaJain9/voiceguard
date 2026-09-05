import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) Layer.
    Computes masked self-attention over spectral and temporal acoustic node embeddings
    to uncover non-local synthetic phase and vocoder artifacts.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2, concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Linear projection weight matrix
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Self-attention mechanism projection vector
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node representations of shape (N, in_features)
            adj: Adjacency matrix of shape (N, N)
        """
        Wh = torch.mm(h, self.W)  # Shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # Broadcast add across rows and columns
        return self.leakyrelu(Wh1 + Wh2.T)


class AASISTBackEnd(nn.Module):
    """
    Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention (AASIST) Backend.
    Models the topological relations between acoustic time frames and spectral frequency bands.
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 128, num_classes: int = 2):
        super(AASISTBackEnd, self).__init__()
        self.gat1 = GraphAttentionLayer(input_dim, hidden_dim, dropout=0.2, alpha=0.2)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=0.2, alpha=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden representations from Wav2Vec2 with shape (Batch, Nodes/Time, Features)
        """
        batch_size, nodes, features = x.shape
        adj = torch.ones(nodes, nodes, device=x.device)

        out_list = []
        for i in range(batch_size):
            h = x[i]
            h = self.gat1(h, adj)
            h = self.gat2(h, adj)
            # Global Average Graph Readout
            h_pooled = torch.mean(h, dim=0)
            out_list.append(h_pooled)

        out = torch.stack(out_list)
        return self.fc(out)


class DeepfakeDetectorModel(nn.Module):
    """
    VoxGuard Hybrid Multilingual Deepfake Detection Architecture.
    
    1. Front-End: Self-Supervised Wav2Vec 2.0 XLS-R (300M parameters)
       Extracts rich cross-lingual phoneme and acoustic representations.
    2. Back-End: AASIST Graph Attention Network (GAT)
       Classifies whether speech exhibits biological vocal tract resonance or neural vocoder synthesis.
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-xls-r-300m", num_classes: int = 2):
        super(DeepfakeDetectorModel, self).__init__()
        print(f"Loading Wav2Vec2 XLS-R Base Model: {model_name}...")
        self.ssl_model = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze low-level CNN feature encoder to preserve general acoustic representations
        self.ssl_model.feature_extractor._freeze_parameters()

        # AASIST GAT Backend (XLS-R 300M produces 1024-dimensional hidden representations)
        self.backend = AASISTBackEnd(input_dim=1024, hidden_dim=128, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw normalized waveform tensor of shape (Batch, Samples)
        Returns:
            logits: Output classification logits of shape (Batch, 2)
        """
        outputs = self.ssl_model(x)
        last_hidden_state = outputs.last_hidden_state  # Shape: (Batch, Time, 1024)
        logits = self.backend(last_hidden_state)
        return logits
