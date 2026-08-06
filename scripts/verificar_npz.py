"""
Verifica quais palavras (classes) existem de fato dentro de um .npz de
dataset, e quantas amostras cada uma tem. Roda direto contra o arquivo que
a Pessoa 2 te mandou -- nao precisa de mais nada dela.

Uso:
    python verificar_npz.py caminho/para/dataset_completo.npz
"""
import sys
from collections import Counter

import numpy as np


def verificar(caminho: str) -> None:
    dados = np.load(caminho, allow_pickle=True)

    print(f"Chaves encontradas no .npz: {list(dados.keys())}\n")

    if "sequencias" not in dados or "rotulos" not in dados:
        print("ATENCAO: esperava as chaves 'sequencias' e 'rotulos' -- confirme com a Pessoa 2 se o formato bate com o Contrato B.")
        return

    sequencias = dados["sequencias"]
    rotulos = dados["rotulos"]

    print(f"Shape de 'sequencias': {sequencias.shape}  (deveria ser (N, 30, 260))")
    print(f"Shape de 'rotulos':    {rotulos.shape}\n")

    contagem = Counter(rotulos.tolist())
    classes_ordenadas = sorted(contagem.keys())

    print(f"Total de classes encontradas: {len(classes_ordenadas)}\n")
    print(f"{'Palavra':<20} {'Qtd amostras':>12}")
    print("-" * 34)
    for classe in classes_ordenadas:
        alerta = "  <- ATENCAO: menos de 3 amostras" if contagem[classe] < 3 else ""
        print(f"{classe:<20} {contagem[classe]:>12}{alerta}")

    print(f"\nLista pronta para colar em vocabulario.json:")
    print(classes_ordenadas)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python verificar_npz.py caminho/para/dataset.npz")
        sys.exit(1)
    verificar(sys.argv[1])