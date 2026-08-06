"""
Loop de treino (v4)
  - split de VALIDACAO por SINALIZADOR (GroupShuffleSplit usando
    sinalizadores_train), nao mais aleatorio por video -- assim a validacao
    tambem mede generalizacao pra gente nova, como o teste
  - o conjunto de teste (X_test/y_test, holdout de sinalizador desconhecido)
    e usado UMA UNICA VEZ, no final, so para reportar o numero real
  - augmentation com espelhamento lateral (robustez a mao dominante),
    ruido e mascaramento de frames
  - gradient clipping, class weights e scheduler de LR

Hiperparametros continuam os travados na Secao 3.5 do CONTRATOS.md:
  - Loss: CrossEntropyLoss / Otimizador: Adam (lr=1e-3, weight_decay=1e-5)
  - Batch size: 16 / Ate 100 epocas, early stopping (paciencia 10)

Uso:
    python -m src.modelo.treino
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.modelo.arquiteturas import ClassificadorLSTM
from src.modelo.dataset import LibrasLandmarksDataset, carregar_npz

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR_SAIDA = os.path.join(RAIZ, "data", "processed")

LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
MAX_EPOCAS = 100
PACIENCIA = 10
VAL_SIZE = 0.2  # fracao do treino separada para validacao
GRAD_CLIP_NORM = 5.0


def _separar_treino_validacao(X, y, sinalizadores):
    """
    Separa treino/validacao. Se houver info de sinalizador, faz o split por
    GRUPO (sinalizador) -- a validacao usa pessoas que nao aparecem no
    treino, igual o holdout de teste. Sem essa info, cai pra um split
    aleatorio por amostra (pior: superestima a generalizacao).
    """
    if sinalizadores is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=42)
        idx_train, idx_val = next(splitter.split(X, y, groups=sinalizadores))
    else:
        print(
            "AVISO: sinalizadores_train nao encontrado no .npz -- split de "
            "validacao sera aleatorio por amostra, nao por sinalizador "
            "(a acc de validacao vai superestimar a generalizacao real)."
        )
        idx_train, idx_val = train_test_split(
            np.arange(len(y)), test_size=VAL_SIZE, stratify=y, random_state=42
        )
    return X[idx_train], y[idx_train], X[idx_val], y[idx_val]


def treinar() -> None:
    os.makedirs(DIR_SAIDA, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {dispositivo}")

    dados = carregar_npz()
    classes = dados["classes"]
    print(f"Classes carregadas: {len(classes)} -> {classes}")

    X_train_full, y_train_full = dados["X_train"], dados["y_train"]
    sinalizadores_train_full = dados.get("sinalizadores_train")

    X_train, y_train, X_val, y_val = _separar_treino_validacao(
        X_train_full, y_train_full, sinalizadores_train_full
    )
    print(f"Treino: {len(y_train)} | Validacao: {len(y_val)} | Teste (holdout): {len(dados['y_test'])}")

    ds_treino = LibrasLandmarksDataset(X_train, y_train, fit_scaler=True, augment=True)
    ds_val = LibrasLandmarksDataset(X_val, y_val, scaler=ds_treino.scaler, fit_scaler=False, augment=False)
    ds_teste = LibrasLandmarksDataset(
        dados["X_test"], dados["y_test"], scaler=ds_treino.scaler, fit_scaler=False, augment=False
    )

    dl_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    dl_teste = DataLoader(ds_teste, batch_size=BATCH_SIZE, shuffle=False)

    n_features = X_train.shape[-1]
    modelo = ClassificadorLSTM(n_features=n_features, n_classes=len(classes)).to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    pesos_classe = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    pesos_classe = torch.tensor(pesos_classe, dtype=torch.float32).to(dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos_classe)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", factor=0.5, patience=4
    )

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
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP_NORM)
            otimizador.step()
            perda_total += perda.item() * x.size(0)
        perda_media = perda_total / len(ds_treino)

        acc_val = avaliar_rapido(modelo, dl_val, dispositivo)
        scheduler.step(acc_val)
        lr_atual = otimizador.param_groups[0]["lr"]
        print(f"Epoca {epoca:03d} | perda_treino={perda_media:.4f} | acc_val={acc_val:.4f} | lr={lr_atual:.2e}")

        if acc_val > melhor_acc_val:
            melhor_acc_val = acc_val
            epocas_sem_melhora = 0
            torch.save(
                {
                    "state_dict": modelo.state_dict(),
                    "n_features": n_features,
                    "vocabulario": classes,  # ORDEM preservada -- fonte de verdade
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

    print(f"Treino finalizado. Melhor acc de VALIDACAO (por sinalizador) = {melhor_acc_val:.4f}")

    # teste final -- usado UMA UNICA VEZ, so para reportar o numero real
    checkpoint = torch.load(os.path.join(DIR_SAIDA, "modelo_melhor.pt"), map_location=dispositivo)
    modelo.load_state_dict(checkpoint["state_dict"])
    acc_teste = avaliar_rapido(modelo, dl_teste, dispositivo)
    print(f"Acuracia final no TESTE (holdout de sinalizador desconhecido) = {acc_teste:.4f}")


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