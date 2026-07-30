import os
import sys

# Adiciona a pasta raiz (libras-translator) ao caminho do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import time
from src.captura.captura import capturar_frames
from src.captura.preprocessamento import preprocessar
from src.captura.segmentacao_classica import segmentar_pele_ycbcr

def main():
    print("Inicializando webcam...")
    
    # Cria o diretório de destino caso não exista
    mock_dir = os.path.join(os.path.dirname(__file__), "..", "data", "mocks")
    os.makedirs(mock_dir, exist_ok=True)
    
    video_path = os.path.join(mock_dir, "frame_exemplo.mp4")
    
    # Configura o gravador usando OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    frames_capturados = 0
    segundos_desejados = 10
    total_frames = segundos_desejados * 30
    
    print(f"\nAperte a barra de espaço na janela para começar a gravar os {segundos_desejados} segundos.")
    print("Aperte 'q' para sair a qualquer momento.")

    iniciar_gravacao = False

    for frame in capturar_frames(0):
        # Gera e aplica os processos clássicos apenas para visualização de debug
        mascara = segmentar_pele_ycbcr(frame)
        mascara_bgr = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
        
        # Mostra o preview
        img_empilhada = cv2.hconcat([frame, mascara_bgr])
        cv2.imshow("Preview - Pressione ESPACO para gravar ou Q para sair", img_empilhada)
        
        # Captura as teclas pressionadas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Abortando...")
            break
        elif key == ord(' '):
            if not iniciar_gravacao:
                iniciar_gravacao = True
                print("\nGRAVANDO! Faça alguns sinais para a câmera...")
        
        if iniciar_gravacao:
            out.write(frame)
            frames_capturados += 1
            
            progresso = (frames_capturados / total_frames) * 100
            print(f"Progresso: {progresso:.1f}% ({frames_capturados}/{total_frames})", end='\r')
            
            if frames_capturados >= total_frames:
                print("\n\nGravação concluída com sucesso!")
                break
                
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Arquivo salvo em: {video_path}")

if __name__ == "__main__":
    main()
