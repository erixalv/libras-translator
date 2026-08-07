import json
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.landmarks.normalizacao import espelhar_vetor

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMINHO_VOCAB = os.path.join(RAIZ, "vocabulario.json")

CAMINHO_DADOS = os.path.join(RAIZ, "data", "processed", "dataset_final.npz")


def carregar_vocabulario() -> list[str]:
    with open(CAMINHO_VOCAB, "r", encoding="utf-8") as f:
        return json.load(f)["sinais"]


def carregar_npz(caminho: str = CAMINHO_DADOS) -> dict:
    dados = np.load(caminho, allow_pickle=True)

    classes_ordem_original = dados["classes"].tolist()
    vocabulario = carregar_vocabulario()

    if sorted(classes_ordem_original) != sorted(vocabulario):
        faltando_no_vocab = set(classes_ordem_original) - set(vocabulario)
        faltando_no_dataset = set(vocabulario) - set(classes_ordem_original)
        msg = "vocabulario.json e as 'classes' do .npz NAO BATEM.\n"
        if faltando_no_vocab:
            msg += f"  Existem no dataset mas nao no vocabulario.json: {faltando_no_vocab}\n"
        if faltando_no_dataset:
            msg += f"  Existem no vocabulario.json mas nao no dataset: {faltando_no_dataset}\n"
        raise ValueError(msg)

    return {
        "X_train": dados["X_train"],
        "y_train": dados["y_train"],
        "sinalizadores_train": dados.get("sinalizadores_train"),
        "X_test": dados["X_test"],
        "y_test": dados["y_test"],
        "sinalizadores_test": dados.get("sinalizadores_test"),
        "classes": classes_ordem_original,  # ORDEM PRESERVADA -- e o mapa indice -> nome
    }


def espelhar_sequencias(X: np.ndarray) -> np.ndarray:
    """
    Espelha lateralmente um lote de sequencias (N, T, N_FEATURES), frame a
    frame, via espelhar_vetor (inverte eixo x, troca pares esq/dir da pose e
    os blocos de mao).

    Precisa ser chamado ANTES do StandardScaler: a inversao de x so tem
    significado geometrico em coordenadas normalizadas cruas (centro dos
    ombros), nao em z-score -- depois de escalado, negar o valor nao
    corresponde mais a espelhar a coordenada original.
    """
    n, n_frames, n_features = X.shape
    espelhado = np.empty_like(X)
    for i in range(n):
        for t in range(n_frames):
            espelhado[i, t] = espelhar_vetor(X[i, t])
    return espelhado


def _augmentar_sequencia(x: np.ndarray, ruido_std: float = 0.01, prob_mascara: float = 0.1) -> np.ndarray:
    """Augmentation leve para sequencias de landmarks, aplicada so no treino:
    - ruido gaussiano pequeno em cada coordenada (simula jitter de deteccao)
    - mascaramento aleatorio de alguns frames, zerando-os (forca o modelo a
      nao depender de um frame especifico da sequencia)
    """
    x_aug = x.copy()
    x_aug = x_aug + np.random.normal(0, ruido_std, size=x_aug.shape).astype(x_aug.dtype)

    n_frames = x_aug.shape[0]
    mascara = np.random.rand(n_frames) < prob_mascara
    x_aug[mascara] = 0.0

    return x_aug


class LibrasLandmarksDataset(Dataset):

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
        augment: bool = False,
    ):
        n, n_frames, n_features = X.shape
        flat = X.reshape(-1, n_features)

        if fit_scaler:
            self.scaler = StandardScaler()
            flat = self.scaler.fit_transform(flat)
        else:
            assert scaler is not None, "Forneca um scaler ja ajustado quando fit_scaler=False"
            self.scaler = scaler
            flat = self.scaler.transform(flat)

        self.X = flat.reshape(n, n_frames, n_features).astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment:
            x = _augmentar_sequencia(x)
        x = torch.from_numpy(x)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y