"""
Gera sequencias sinteticas de landmarks
"""
import json
import os

import numpy as np

N_FRAMES = 30
N_FEATURES = 258  # 33*4 (pose) + 21*3*2 (maos) -- ver CONTRATOS.md, Contrato B
AMOSTRAS_POR_CLASSE = 40
SEED = 42

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_VOCAB = os.path.join(RAIZ, "vocabulario.json")
SAIDA_DIR = os.path.join(RAIZ, "data", "mocks")


def gerar_sequencia_sintetica(rng: np.random.Generator) -> np.ndarray:
    """Gera uma sequencia (N_FRAMES, N_FEATURES) com padrao suave, nao
    puramente aleatorio -- assim o modelo tem algo minimamente aprendivel
    mesmo com dado falso, o que ajuda a validar se o pipeline de treino
    esta funcionando (a acuracia deve subir acima do acaso)."""
    base = rng.normal(loc=0.0, scale=1.0, size=(N_FEATURES,))
    ruido = rng.normal(loc=0.0, scale=0.05, size=(N_FRAMES, N_FEATURES))
    tendencia = np.linspace(0, rng.uniform(-0.3, 0.3), N_FRAMES)[:, None]
    return base[None, :] + tendencia + ruido


def main() -> None:
    rng = np.random.default_rng(SEED)

    with open(CAMINHO_VOCAB, "r", encoding="utf-8") as f:
        vocabulario = json.load(f)["sinais"]

    os.makedirs(SAIDA_DIR, exist_ok=True)

    sequencias = []
    rotulos = []
    for classe in vocabulario:
        for i in range(AMOSTRAS_POR_CLASSE):
            seq = gerar_sequencia_sintetica(rng)
            sequencias.append(seq)
            rotulos.append(classe)

    sequencias = np.stack(sequencias).astype(np.float32)  # (N, 30, 258)
    rotulos = np.array(rotulos)

    caminho_saida = os.path.join(SAIDA_DIR, "landmarks_exemplo.npz")
    np.savez_compressed(caminho_saida, sequencias=sequencias, rotulos=rotulos)

    print(f"Gerado: {caminho_saida}")
    print(f"  shape sequencias: {sequencias.shape}")
    print(f"  classes: {len(vocabulario)} -> {vocabulario}")
    print(f"  amostras por classe: {AMOSTRAS_POR_CLASSE}")


if __name__ == "__main__":
    main()
