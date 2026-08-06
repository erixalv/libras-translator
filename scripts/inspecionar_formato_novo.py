"""
Inspeciona o formato alternativo que a Pessoa 2 entregou (com split
treino/teste e sinalizadores ja embutidos), para decidir como adaptar.
"""
import sys
import numpy as np
from collections import Counter


def inspecionar(caminho: str) -> None:
    dados = np.load(caminho, allow_pickle=True)
    print(f"Chaves: {list(dados.keys())}\n")

    for chave in dados.keys():
        arr = dados[chave]
        print(f"{chave:<20} shape={arr.shape}  dtype={arr.dtype}")

    print()
    if "classes" in dados:
        classes = dados["classes"]
        print(f"'classes' contem {len(classes)} itens: {classes.tolist()[:10]}{'...' if len(classes) > 10 else ''}\n")

    if "y_train" in dados:
        contagem_treino = Counter(dados["y_train"].tolist())
        print(f"Distribuicao y_train ({len(contagem_treino)} classes distintas):")
        for k in sorted(contagem_treino.keys())[:10]:
            print(f"  {k}: {contagem_treino[k]}")
        if len(contagem_treino) > 10:
            print(f"  ... e mais {len(contagem_treino) - 10} classes")

    if "sinalizadores_train" in dados:
        sinalizadores = set(dados["sinalizadores_train"].tolist())
        print(f"\nSinalizadores distintos no treino: {sinalizadores}")
    if "sinalizadores_test" in dados:
        sinalizadores_teste = set(dados["sinalizadores_test"].tolist())
        print(f"Sinalizadores distintos no teste: {sinalizadores_teste}")

    if "X_train" in dados:
        print(f"\nExemplo de valores em X_train[0,0,:5]: {dados['X_train'][0,0,:5]}")


if __name__ == "__main__":
    inspecionar(sys.argv[1])