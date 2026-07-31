"""
Script de gravacao guiada para o dataset proprio de Libras.

Percorre automaticamente a lista de palavras (pronomes + Grupo B), pedindo
o numero de repeticoes correto de acordo com o PAPEL da pessoa (treino ou
teste/holdout), e salva cada repeticao como 1 arquivo de video, ja com o
nome padronizado esperado pelo resto do pipeline:

    {SINAL}_{pessoa}_{indice:02d}.mp4

Uso:
    python scripts/gravar_vocabulario.py --pessoa pessoaA --papel treino
    python scripts/gravar_vocabulario.py --pessoa pessoaE --papel teste

Controles durante a gravacao:
    ESPACO -> inicia a gravacao da repeticao atual (grava por conta propria
              ate detectar ~1s de mao parada, ou ate ENTER ser pressionado)
    ENTER  -> encerra a repeticao atual manualmente
    R      -> repete a mesma amostra de novo (descarta a ultima gravacao)
    Q      -> aborta o script inteiro
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2

from src.captura.captura import capturar_frames
from src.captura.segmentacao_classica import segmentar_pele_ycbcr

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAMINHO_CONFIG = os.path.join(os.path.dirname(__file__), "config_gravacao.json")


def carregar_config() -> dict:
    with open(CAMINHO_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def montar_plano_de_gravacao(config: dict, papel: str) -> list[tuple[str, int]]:
    """Retorna lista de (palavra, numero_de_repeticoes) na ordem a gravar."""
    reps = config["repeticoes"][papel]
    plano = [(p, reps["pronomes"]) for p in config["pronomes"]]
    plano += [(p, reps["grupo_b"]) for p in config["grupo_b"]]
    return plano


def caminho_saida(papel: str, palavra: str, pessoa: str, indice: int) -> str:
    subpasta = "treino" if papel == "treino" else "teste_holdout"
    pasta = os.path.join(RAIZ, "data", "raw_gravacoes", subpasta, palavra)
    os.makedirs(pasta, exist_ok=True)
    nome_arquivo = f"{palavra}_{pessoa}_{indice:02d}.mp4"
    return os.path.join(pasta, nome_arquivo)


def gravar_uma_repeticao(caminho_arquivo: str) -> bool:
    """
    Grava 1 repeticao: mostra preview, espera ESPACO para comecar a gravar,
    grava ate ENTER (fim manual) ou Q (aborta tudo).
    Retorna True se gravou com sucesso, False se o usuario abortou.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(caminho_arquivo, fourcc, 30.0, (640, 480))

    gravando = False
    frames_gravados = 0

    for frame in capturar_frames(0):
        mascara = segmentar_pele_ycbcr(frame)
        mascara_bgr = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
        preview = cv2.hconcat([frame, mascara_bgr])

        cor_status = (0, 0, 255) if not gravando else (0, 255, 0)
        texto_status = "ESPACO para gravar" if not gravando else f"GRAVANDO ({frames_gravados} frames) - ENTER p/ finalizar"
        cv2.putText(preview, texto_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status, 2)
        cv2.imshow("Gravacao de vocabulario - Q para abortar tudo", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            out.release()
            cv2.destroyAllWindows()
            return False

        if key == ord(" ") and not gravando:
            gravando = True

        if gravando:
            out.write(frame)
            frames_gravados += 1

        if gravando and key == 13:  # ENTER
            break

    out.release()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Gravacao guiada de vocabulario")
    parser.add_argument("--pessoa", required=True, help="ex.: pessoaA, pessoaB, pessoaE")
    parser.add_argument("--papel", required=True, choices=["treino", "teste"])
    args = parser.parse_args()

    config = carregar_config()
    plano = montar_plano_de_gravacao(config, args.papel)

    total_palavras = len(plano)
    total_clipes = sum(n for _, n in plano)

    ja_gravados = sum(
        1
        for palavra, n_repeticoes in plano
        for indice in range(1, n_repeticoes + 1)
        if os.path.exists(caminho_saida(args.papel, palavra, args.pessoa, indice))
    )

    print(f"Plano de gravacao para {args.pessoa} ({args.papel}): {total_palavras} palavras, {total_clipes} clipes no total.")
    if ja_gravados > 0:
        print(f"Retomando sessao anterior: {ja_gravados}/{total_clipes} ja gravados, faltam {total_clipes - ja_gravados}.\n")
    else:
        print()

    clipes_feitos = 0
    clipes_pulados = 0
    for palavra, n_repeticoes in plano:
        print(f"\n=== Palavra: {palavra} ({n_repeticoes} repeticoes) ===")
        indice = 1
        while indice <= n_repeticoes:
            destino = caminho_saida(args.papel, palavra, args.pessoa, indice)

            if os.path.exists(destino):
                print(f"  Repeticao {indice}/{n_repeticoes} -- ja existe, pulando ({os.path.basename(destino)})")
                clipes_pulados += 1
                clipes_feitos += 1
                indice += 1
                continue

            print(f"  Repeticao {indice}/{n_repeticoes} -- prepare o sinal '{palavra}' e aperte ESPACO na janela.")
            sucesso = gravar_uma_repeticao(destino)

            if not sucesso:
                print("\nAbortado pelo usuario. Progresso salvo ate aqui.")
                cv2.destroyAllWindows()
                return

            clipes_feitos += 1
            print(f"  Salvo: {destino}  (progresso geral: {clipes_feitos}/{total_clipes})")
            indice += 1

    cv2.destroyAllWindows()
    if clipes_pulados:
        print(f"\nConcluido! {clipes_feitos}/{total_clipes} clipes ({clipes_pulados} ja existiam de uma sessao anterior).")
    else:
        print(f"\nConcluido! {clipes_feitos} clipes gravados para {args.pessoa} ({args.papel}).")


if __name__ == "__main__":
    main()