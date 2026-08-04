"""
Organiza as gravacoes proprias (pastas "treino", "treino 2", "treino 3",
"treino 4", "teste", ...) na convencao que dataset_builder.py espera:

    data/raw/GRAVACOES/<SINAL>/<arquivo>_<sinalizador>.mp4

Nome de arquivo original: "<SINAL_MAIUSCULO>_pessoa<X>_<NN>.mp4"
    ex: "OBRIGADO_pessoaC_04.mp4" -> sinal="OBRIGADO", pessoa="C", rep="04"

O nome da pasta original (ex.: "OBRIGADO", "AGUA") usa a convencao do
Contrato 0 (caixa alta, sem acento). Isso e traduzido para o nome EXATO
que ja existe em data/raw/VLIBRASIL/ (ex.: "Obrigado", "Agua" -> "Água"),
senao essas gravacoes viram uma classe DUPLICADA em vez de somar amostra
as palavras que ja existem. EU e VOCE nao existem no V-LIBRASIL — viram
classes novas ("Eu", "Você").

Uso:
    python scripts/organizar_gravacoes.py
    python scripts/organizar_gravacoes.py --pastas "treino" "treino 2" "teste"
"""

import argparse
import re
from pathlib import Path

# nome da pasta original -> nome EXATO da classe (verificado contra data/raw/VLIBRASIL)
MAPA_SINAL = {
    "QUER": "Quer",
    "AJUDAR": "Ajudar",
    "GOSTAR": "Gostar",
    "COMER": "Comer",
    "IR": "Ir",
    "SIM": "Sim",
    "NAO": "Não",
    "OI": "Oi",
    "OBRIGADO": "Obrigado",
    "POR_FAVOR": "Por favor",
    "AGUA": "Água",
    "CASA": "Casa",
    "DORMIR": "Dormir",
    "VER": "Ver",
    "AMIGO": "Amigo",
    "EU": "Eu",       # novo, nao existe no V-LIBRASIL
    "VOCE": "Você",   # novo, nao existe no V-LIBRASIL
}

PADRAO_NOME = re.compile(r"^(?P<sinal>.+?)_pessoa(?P<pessoa>[A-Za-z0-9]+)_(?P<rep>\d+)\.mp4$")


def organizar(pastas: list[Path], destino: Path) -> dict:
    criados = pulados_nome = pulados_sinal_desconhecido = pulados_existente = 0

    for pasta in pastas:
        if not pasta.exists():
            print(f"aviso: pasta {pasta} nao existe, pulando")
            continue

        for caminho in sorted(pasta.glob("*/*.mp4")):
            m = PADRAO_NOME.match(caminho.name)
            if not m:
                pulados_nome += 1
                continue

            sinal_bruto, pessoa, rep = m["sinal"], m["pessoa"], m["rep"]
            sinal = MAPA_SINAL.get(sinal_bruto)
            if sinal is None:
                pulados_sinal_desconhecido += 1
                print(f"  sinal desconhecido: {sinal_bruto!r} ({caminho})")
                continue

            pasta_sinal = destino / sinal
            pasta_sinal.mkdir(parents=True, exist_ok=True)
            destino_arquivo = pasta_sinal / f"rep{rep}_pessoa{pessoa}.mp4"

            if destino_arquivo.exists() or destino_arquivo.is_symlink():
                pulados_existente += 1
                continue

            destino_arquivo.symlink_to(caminho.resolve())
            criados += 1

    return {
        "criados": criados,
        "pulados_nome": pulados_nome,
        "pulados_sinal_desconhecido": pulados_sinal_desconhecido,
        "pulados_existente": pulados_existente,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pastas",
        nargs="+",
        default=["treino", "treino 2", "treino 3", "treino 4", "teste"],
    )
    ap.add_argument("--destino", type=Path, default=Path("data/raw/GRAVACOES"))
    args = ap.parse_args()

    resultado = organizar([Path(p) for p in args.pastas], args.destino)

    print(f"\nlinks criados:                  {resultado['criados']}")
    print(f"pulados (nome fora do padrao):  {resultado['pulados_nome']}")
    print(f"pulados (sinal desconhecido):   {resultado['pulados_sinal_desconhecido']}")
    print(f"pulados (ja existia):           {resultado['pulados_existente']}")


if __name__ == "__main__":
    main()
