"""
Classificador LSTM bidirecional para landmarks de mao (LIBRAS)
  - LSTM bidirecional, 2 camadas (arquitetura travada na Secao 3.5 do
    CONTRATOS.md), hidden_size=48 e dropout mais forte -- reduzidos em
    relacao ao hidden_size=128/dropout=0.3 originais depois de medir
    overfitting severo (~99% treino vs ~35-40% em sinalizador nunca visto,
    com apenas 12 sinalizadores distintos no dataset -- ver discussao do
    dia em que isso foi diagnosticado). Continua sendo uma LSTM bidirecional
    de 2 camadas, so com capacidade calibrada pro tamanho real do dataset.
  - Dropout adicional antes da camada linear de saida (regularizacao extra)
  - Linear(hidden_size*2, n_classes) na saida (bidirecional)
"""
import torch
import torch.nn as nn


class ClassificadorLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 260,
        hidden_size: int = 48,
        n_layers: int = 2,
        dropout: float = 0.4,
        dropout_saida: float = 0.4,
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
        self.dropout_saida = nn.Dropout(dropout_saida)
        self.classificador = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        ultima_fwd = h_n[-2]  # ultima camada, direcao forward
        ultima_bwd = h_n[-1]  # ultima camada, direcao backward
        representacao = torch.cat([ultima_fwd, ultima_bwd], dim=1)
        representacao = self.dropout_saida(representacao)
        return self.classificador(representacao)