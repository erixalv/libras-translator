import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.modelo.arquiteturas import ClassificadorLSTM
from src.modelo.dataset import LibrasLandmarksDataset, carregar_dados_brutos, carregar_vocabulario

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_SAIDA = os.path.join(RAIZ, "data", "processed")

LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
MAX_EPOCAS = 100
PACIENCIA = 10


def treinar() -> None:
    os.makedirs(DIR_SAIDA, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {dispositivo}")

    vocabulario = carregar_vocabulario()
    sequencias, rotulos = carregar_dados_brutos()

    seq_treino, seq_val, rot_treino, rot_val = train_test_split(
        sequencias, rotulos, test_size=0.2, random_state=42, stratify=rotulos
    )

    ds_treino = LibrasLandmarksDataset(seq_treino, rot_treino, vocabulario, fit_scaler=True)
    ds_val = LibrasLandmarksDataset(seq_val, rot_val, vocabulario, scaler=ds_treino.scaler, fit_scaler=False)

    dl_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    modelo = ClassificadorLSTM(n_features=sequencias.shape[-1], n_classes=len(vocabulario)).to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterio = nn.CrossEntropyLoss()

    melhor_acc_val = 0.0
    epocas_sem_melhora = 0

    for epoca in range(1, MAX_EPOCAS + 1):
        modelo.train()
        perda_total = 0.0
        for x, y in dl_treino:
            x, y = x.to(dispositivo), y.to(dispositivo)
            otimizador.zero_grad()
            logits = modelo(x)
            perda = criterio(logits, y)
            perda.backward()
            otimizador.step()
            perda_total += perda.item() * x.size(0)
        perda_media = perda_total / len(ds_treino)

        acc_val = avaliar_rapido(modelo, dl_val, dispositivo)
        print(f"Epoca {epoca:03d} | perda_treino={perda_media:.4f} | acc_val={acc_val:.4f}")

        if acc_val > melhor_acc_val:
            melhor_acc_val = acc_val
            epocas_sem_melhora = 0
            torch.save(
                {
                    "state_dict": modelo.state_dict(),
                    "n_features": sequencias.shape[-1],
                    "vocabulario": vocabulario,
                },
                os.path.join(DIR_SAIDA, "modelo_melhor.pt"),
            )
            with open(os.path.join(DIR_SAIDA, "scaler.pkl"), "wb") as f:
                pickle.dump(ds_treino.scaler, f)
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= PACIENCIA:
                print(f"Early stopping na epoca {epoca} (sem melhora ha {PACIENCIA} epocas).")
                break

    print(f"Treino finalizado. Melhor acc_val = {melhor_acc_val:.4f}")


def avaliar_rapido(modelo, dl, dispositivo) -> float:
    modelo.eval()
    corretos, total = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(dispositivo), y.to(dispositivo)
            preds = modelo(x).argmax(dim=1)
            corretos += (preds == y).sum().item()
            total += y.size(0)
    return corretos / total if total > 0 else 0.0


if __name__ == "__main__":
    treinar()
