import json
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMINHO_VOCAB = os.path.join(RAIZ, "vocabulario.json")

# TROCAR PELO DATASET REAL QUANDO FOR ENTREGUE
CAMINHO_DADOS = os.path.join(RAIZ, "data", "mocks", "landmarks_exemplo.npz")


def carregar_vocabulario() -> list[str]:
    with open(CAMINHO_VOCAB, "r", encoding="utf-8") as f:
        return json.load(f)["sinais"]


class LibrasLandmarksDataset(Dataset):
    """
    Espera um .npz com:
      - 'sequencias': array (N, 30, N_FEATURES) float32
      - 'rotulos':    array (N,) de strings, cada uma um item de vocabulario.json

    O StandardScaler e ajustado apenas no conjunto de treino (fit=True) e
    reaproveitado na validacao/inferencia (fit=False), conforme Secao 3.5
    do CONTRATOS.md.
    """

    def __init__(
        self,
        sequencias: np.ndarray,
        rotulos: np.ndarray,
        vocabulario: list[str],
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ):
        self.vocabulario = vocabulario
        self.classe_para_indice = {c: i for i, c in enumerate(vocabulario)}

        n, n_frames, n_features = sequencias.shape
        flat = sequencias.reshape(-1, n_features)

        #transforma cada feature pra ter média 0 e desvio-padrão 1
        if fit_scaler:
            self.scaler = StandardScaler()
            flat = self.scaler.fit_transform(flat)
        else:
            assert scaler is not None, "Forneca um scaler ja ajustado quando fit_scaler=False"
            self.scaler = scaler
            flat = self.scaler.transform(flat)

        self.sequencias = flat.reshape(n, n_frames, n_features).astype(np.float32)
        self.rotulos_indice = np.array([self.classe_para_indice[r] for r in rotulos])

    def __len__(self) -> int:
        return len(self.sequencias)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.sequencias[idx])
        y = torch.tensor(self.rotulos_indice[idx], dtype=torch.long)
        return x, y


def carregar_dados_brutos(caminho: str = CAMINHO_DADOS):
    dados = np.load(caminho, allow_pickle=True)
    return dados["sequencias"], dados["rotulos"]
