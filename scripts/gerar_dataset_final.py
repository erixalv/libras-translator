"""
Filtra o dataset completo (V-LIBRASIL + MINDS-Libras + Gravacoes proprias)
para conter APENAS as palavras que tem cobertura reforcada por gravacao
propria: as 20 palavras do MINDS-Libras + as 17 palavras gravadas pelo
grupo (15 que ja existiam no V-LIBRASIL + Eu/Você, que sao novas).

Le data/processed/dataset_vlibrasil.npz (V-LIBRASIL + GRAVACOES, ja que
ambos vivem em data/raw/) e data/processed/dataset_minds.npz, filtra pelas
classes selecionadas, remapeia os indices e junta tudo em
data/processed/dataset_final.npz. Tambem reescreve vocabulario.json para
listar so essas classes.

Uso:
    python scripts/gerar_dataset_final.py
"""

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_VLIBRASIL = RAIZ / "data/processed/dataset_vlibrasil.npz"
CAMINHO_MINDS = RAIZ / "data/processed/dataset_minds.npz"
CAMINHO_SAIDA = RAIZ / "data/processed/dataset_final.npz"
CAMINHO_VOCAB = RAIZ / "vocabulario.json"

# palavras gravadas pelo grupo (data/raw/GRAVACOES) — ver scripts/organizar_gravacoes.py
PALAVRAS_GRAVACAO_PROPRIA = {
    "Quer", "Ajudar", "Gostar", "Comer", "Ir", "Sim", "Não", "Oi",
    "Obrigado", "Por favor", "Água", "Casa", "Dormir", "Ver", "Amigo",
    "Eu", "Você",
}


def _filtrar_e_remapear(
    dados: dict, classes_origem: list[str], selecionadas: list[str], indice_final: dict[str, int]
) -> dict:
    """Filtra X/y/sinalizador de treino e teste para as classes selecionadas,
    remapeando y para o indice em `indice_final`."""
    resultado = {}
    for split in ("train", "test"):
        X, y, sinalizador = (
            dados[f"X_{split}"],
            dados[f"y_{split}"],
            dados[f"sinalizadores_{split}"],
        )
        mascara = np.array([classes_origem[i] in selecionadas for i in y])
        y_str = np.array([classes_origem[i] for i in y])[mascara]
        resultado[f"X_{split}"] = X[mascara]
        resultado[f"y_{split}"] = np.array([indice_final[c] for c in y_str], dtype=np.int64)
        resultado[f"sinalizadores_{split}"] = sinalizador[mascara]
    return resultado


def gerar() -> None:
    vocab = json.loads(CAMINHO_VOCAB.read_text(encoding="utf-8"))
    origem_por_sinal = vocab["origem_por_sinal"]

    palavras_minds = {p for p, o in origem_por_sinal.items() if "MINDS" in o}
    selecionadas = sorted(palavras_minds | PALAVRAS_GRAVACAO_PROPRIA)
    indice_final = {c: i for i, c in enumerate(selecionadas)}

    dv = np.load(CAMINHO_VLIBRASIL, allow_pickle=True)
    dm = np.load(CAMINHO_MINDS, allow_pickle=True)
    classes_v = list(dv["classes"])
    classes_m = list(dm["classes"])

    parte_v = _filtrar_e_remapear(dv, classes_v, selecionadas, indice_final)
    parte_m = _filtrar_e_remapear(dm, classes_m, selecionadas, indice_final)

    saida = {
        "X_train": np.concatenate([parte_v["X_train"], parte_m["X_train"]]),
        "y_train": np.concatenate([parte_v["y_train"], parte_m["y_train"]]),
        "sinalizadores_train": np.concatenate(
            [parte_v["sinalizadores_train"], parte_m["sinalizadores_train"]]
        ),
        "X_test": np.concatenate([parte_v["X_test"], parte_m["X_test"]]),
        "y_test": np.concatenate([parte_v["y_test"], parte_m["y_test"]]),
        "sinalizadores_test": np.concatenate(
            [parte_v["sinalizadores_test"], parte_m["sinalizadores_test"]]
        ),
        "classes": np.array(selecionadas),
    }

    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CAMINHO_SAIDA, **saida)

    print(f"dataset final salvo em {CAMINHO_SAIDA}")
    print(f"classes: {len(selecionadas)}")
    print(f"treino: {saida['X_train'].shape[0]} amostras")
    print(f"teste:  {saida['X_test'].shape[0]} amostras")

    novo_origem_por_sinal = {}
    for palavra in selecionadas:
        origem_atual = origem_por_sinal.get(palavra, "")
        if palavra in PALAVRAS_GRAVACAO_PROPRIA and "Gravação própria" not in origem_atual:
            partes = [p for p in [origem_atual, "Gravação própria (data/raw/GRAVACOES)"] if p]
            novo_origem_por_sinal[palavra] = ", ".join(partes)
        else:
            novo_origem_por_sinal[palavra] = origem_atual

    vocab_final = {
        "versao": vocab["versao"] + 1,
        "origem": (
            "Subconjunto filtrado: MINDS-Libras (20 sinais, Kaggle j0aopsantos/minds-libras) "
            "+ V-LIBRASIL/Gravação própria (15 sinais com gravação própria adicional, "
            "data/raw/GRAVACOES) + pronomes Eu/Você (gravação própria)"
        ),
        "sinais": selecionadas,
        "origem_por_sinal": novo_origem_por_sinal,
    }
    CAMINHO_VOCAB.write_text(
        json.dumps(vocab_final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"vocabulario.json reescrito com {len(selecionadas)} sinais (versao {vocab_final['versao']})")


if __name__ == "__main__":
    gerar()
