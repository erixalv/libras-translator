import torch
import torch.nn as nn


class ClassificadorLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 258,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 10,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classificador = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 30, n_features)
        saida, (h_n, _) = self.lstm(x)
        # h_n: (n_layers*2, batch, hidden_size) -- pega a ultima camada, ambas direcoes
        ultima_fwd = h_n[-2]
        ultima_bwd = h_n[-1]
        representacao = torch.cat([ultima_fwd, ultima_bwd], dim=1)  # (batch, hidden*2)
        return self.classificador(representacao)  # (batch, n_classes) -- logits
