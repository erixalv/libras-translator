import sys
import numpy as np
from collections import defaultdict

CLASSES_SUSPEITAS = ["Banheiro", "Filho", "Maca", "Você", "Água", "Ir"]


def diagnosticar(caminho: str) -> None:
    dados = np.load(caminho, allow_pickle=True)
    classes = dados["classes"].tolist()

    for split in ("train", "test"):
        y = dados[f"y_{split}"]
        sinalizadores = dados[f"sinalizadores_{split}"]

        print(f"\n{'='*60}\nSPLIT: {split}\n{'='*60}")
        for classe_suspeita in CLASSES_SUSPEITAS:
            if classe_suspeita not in classes:
                print(f"\n'{classe_suspeita}': nao esta na lista de classes deste dataset.")
                continue
            idx_classe = classes.index(classe_suspeita)
            mascara = y == idx_classe
            sinalizadores_dessa_classe = sinalizadores[mascara]

            contagem = defaultdict(int)
            for s in sinalizadores_dessa_classe:
                contagem[s] += 1

            print(f"\n'{classe_suspeita}' (indice {idx_classe}) -- {mascara.sum()} amostras:")
            for sinalizador, qtd in sorted(contagem.items()):
                print(f"    {sinalizador}: {qtd}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python diagnostico_rotulos.py caminho/para/dataset_final.npz")
        sys.exit(1)
    diagnosticar(sys.argv[1])