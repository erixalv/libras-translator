import cv2
import os
import glob
from pathlib import Path
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_ORIGEM = os.path.join(RAIZ, "data", "raw_gravacoes")
PASTA_DESTINO = os.path.join(RAIZ, "data", "raw_gravacoes_trim")

def get_motion_energy(video_path):
    cap = cv2.VideoCapture(video_path)
    energies = []
    ret, prev_frame = cap.read()
    if not ret: return []
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # Conta quantos pixels mudaram
        energy = np.sum(thresh) / 255.0
        energies.append(energy)
        prev_gray = gray
    cap.release()
    return energies

def trim_video(video_path, out_path):
    energies = get_motion_energy(video_path)
    if not energies or len(energies) < 10: 
        return False
        
    max_e = max(energies)
    if max_e < 100: # Quase nenhumm movimento no video todo
        return False
        
    # Limiar: 15% do movimento maximo do video
    threshold = max_e * 0.15 
    
    start_frame = 0
    end_frame = len(energies) - 1
    
    # Encontra o inicio do movimento real
    for i, e in enumerate(energies):
        if e > threshold:
            start_frame = max(0, i - 5) # 5 frames de folga (suavidade)
            break
            
    # Encontra o final do movimento real
    for i in range(len(energies)-1, -1, -1):
        if energies[i] > threshold:
            end_frame = min(len(energies) - 1, i + 5)
            break
            
    # Se o corte ficou bizarro ou muito curto, ignora e usa o video todo
    if end_frame - start_frame < 10:
        start_frame = 0
        end_frame = len(energies)
        
    # Se nao precisar cortar muito, apenas copia
    if start_frame == 0 and end_frame >= len(energies) - 2:
        start_frame = 0
        end_frame = len(energies)
        
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    idx = 0
    frames_salvos = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if start_frame <= idx <= end_frame:
            out.write(frame)
            frames_salvos += 1
        idx += 1
        
    cap.release()
    out.release()
    
    # Retorna estatisticas
    return (len(energies) + 1, frames_salvos)

def main():
    print("Iniciando purificacao dos videos (Remocao de frames parados)...")
    
    videos = glob.glob(os.path.join(PASTA_ORIGEM, "**", "*.mp4"), recursive=True)
    if not videos:
        print(f"Nenhum video encontrado em {PASTA_ORIGEM}")
        return
        
    total_origem = 0
    total_recortado = 0
    arquivos_processados = 0
    
    for video in videos:
        # Mantem a estrutura de pastas originais (ex: treino/Agua/video.mp4)
        caminho_relativo = os.path.relpath(video, PASTA_ORIGEM)
        out_path = os.path.join(PASTA_DESTINO, caminho_relativo)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        resultado = trim_video(video, out_path)
        if resultado:
            frames_originais, frames_novos = resultado
            total_origem += frames_originais
            total_recortado += frames_novos
            arquivos_processados += 1
            if arquivos_processados % 20 == 0:
                print(f"[{arquivos_processados}/{len(videos)}] Processado: {caminho_relativo} ({frames_originais} -> {frames_novos} frames)")
                
    if arquivos_processados > 0:
        reducao = 100 - (total_recortado / total_origem * 100)
        print("\n=== CONCLUIDO ===")
        print(f"Videos processados: {arquivos_processados}")
        print(f"Total de frames ANTES: {total_origem}")
        print(f"Total de frames AGORA: {total_recortado}")
        print(f"Reducao de silêncio (Lixo removido): {reducao:.1f}%")
        print("\nPROXIMO PASSO: Apague a pasta 'data/raw_gravacoes' original e renomeie 'data/raw_gravacoes_trim' para 'data/raw_gravacoes'.")

if __name__ == "__main__":
    main()
