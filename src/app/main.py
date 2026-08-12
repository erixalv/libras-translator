"""
Interface Principal Streamlit (Pessoa 5).
Tradutor Automático de Libras -> Português.

Para rodar este app:
    streamlit run src/app/main.py
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Garante inclusão do diretório raiz no PYTHONPATH
RAIZ = str(Path(__file__).resolve().parents[2])
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from src.app.pipeline_integrador import PipelineIntegrador


# Configuration de página Streamlit
st.set_page_config(
    page_title="Tradutor de Libras -> Português",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada (Design Dark/Moderno)
st.markdown("""
<style>
    /* Estilos globais */
    .main {
        background-color: #0e1117;
    }
    
    /* Cartões de métrica */
    .metric-card {
        background: linear-gradient(135deg, #1e2640 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }

    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .metric-value-frase {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.3;
    }

    .metric-value-glosas {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f59e0b;
        font-family: monospace;
    }

    .badge-mock {
        background-color: #854d0e;
        color: #fef08a;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .badge-real {
        background-color: #166534;
        color: #bbf7d0;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


def inicializar_estado():
    """Inicializa variáveis de estado da sessão Streamlit."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = PipelineIntegrador(modo_mock=False, limiar_confianca=0.60, mostrar_landmarks=True)
    if "executando" not in st.session_state:
        st.session_state.executando = False


def main():
    inicializar_estado()

    st.title("🤟 Tradutor Automático de Libras → Português")
    st.caption("Sistema de Reconhecimento em Tempo Real via Visão Computacional e Deep Learning")

    # ==========================================
    # SIDEBAR - CONFIGURAÇÕES E CONTROLES
    # ==========================================
    st.sidebar.header("⚙️ Configurações do App")

    # 1. Seletor de fonte de vídeo
    caminho_mock_video = os.path.join(RAIZ, "data", "mocks", "frame_exemplo.mp4")
    opcoes_video = ["Webcam (Ao Vivo)", f"Vídeo de Exemplo ({os.path.basename(caminho_mock_video)})"]
    
    fonte_selecionada = st.sidebar.selectbox(
        "Fonte de Vídeo:",
        opcoes_video,
        index=0
    )

    # 2. Toggle Modo Mock / Real
    modo_mock = st.sidebar.toggle("Modo Mock (Simulação)", value=False)

    # 2.1 Toggle de visualização dos landmarks (debug: confirma se a mão está sendo captada)
    mostrar_landmarks = st.sidebar.toggle(
        "🖐️ Mostrar landmarks detectados",
        value=True,
        help="Desenha os pontos de pose/mãos que o MediaPipe está captando, direto no vídeo."
    )

    # 3. Slider de Confiança
    confianca_slider = st.sidebar.slider(
        "Limiar de Confiança:",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="Glosas com confiança abaixo deste valor são descartadas."
    )

    # 4. Botão de Reset
    if st.sidebar.button("🔄 Resetar Pipeline", use_container_width=True):
        st.session_state.pipeline.reset()
        st.sidebar.success("Pipeline resetado com sucesso!")

    # Atualiza configurações no pipeline integrador
    st.session_state.pipeline.modo_mock = modo_mock
    st.session_state.pipeline.limiar_confianca = confianca_slider
    st.session_state.pipeline.mostrar_landmarks = mostrar_landmarks

    st.sidebar.divider()
    st.sidebar.markdown("**Status da Integração:**")
    if st.session_state.pipeline.modo_mock:
        st.sidebar.markdown('<span class="badge-mock">MODO MOCK ATIVO</span>', unsafe_allow_html=True)
        if not modo_mock:
            st.sidebar.warning("⚠️ Os módulos reais de IA ainda não estão disponíveis/completos. Mantendo o Modo Mock ativo com segurança.")
    else:
        st.sidebar.markdown('<span class="badge-real">MODO REAL ATIVO</span>', unsafe_allow_html=True)

    # ==========================================
    # PAINEL PRINCIPAL - LAYOUT EM COLUNAS
    # ==========================================
    col_video, col_painel = st.columns([1.6, 1.0])

    with col_video:
        st.subheader("📹 Fluxo de Vídeo com Overlay")
        placeholder_video = st.empty()
        
        # Botões de início e parada
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("▶️ Iniciar Tradução", use_container_width=True, type="primary"):
                st.session_state.executando = True
        with col_btn2:
            if st.button("⏹️ Parar", use_container_width=True):
                st.session_state.executando = False

    with col_painel:
        st.subheader("📊 Painel de Tradução")
        
        placeholder_frase = st.empty()
        placeholder_glosas = st.empty()
        placeholder_inst = st.empty()
        placeholder_topk = st.empty()
        placeholder_status = st.empty()

        # Renderização inicial vazia do painel
        with placeholder_frase.container():
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">💬 Frase Traduzida em Português</div>
                <div class="metric-value-frase">...</div>
            </div>
            """, unsafe_allow_html=True)

        with placeholder_glosas.container():
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">🖐️ Glosas no Buffer Acumulado</div>
                <div class="metric-value-glosas">Nenhuma glosa aceita ainda</div>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # LOOP DE PROCESSAMENTO DE VÍDEO
    # ==========================================
    if st.session_state.executando:
        # Determina a fonte de vídeo (Webcam x Arquivo .mp4)
        if "Webcam" in fonte_selecionada:
            fonte_cv = 0
        else:
            fonte_cv = caminho_mock_video

        cap = cv2.VideoCapture(fonte_cv)

        # Se o arquivo mock não existir, gera frames sintéticos para demonstração
        usar_sintetico = False
        if not cap.isOpened():
            if isinstance(fonte_cv, str):
                st.warning(f"Arquivo mock '{caminho_mock_video}' não encontrado. Gerando frames sintéticos.")
                usar_sintetico = True
            else:
                st.error("Não foi possível acessar a Webcam. Verifique as permissões da câmera.")
                st.session_state.executando = False

        while st.session_state.executando:
            if usar_sintetico:
                # Frame sintético 480x640x3
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Adiciona um gradiente dinâmico simples
                t = int(time.time() * 20) % 255
                cv2.circle(frame, (320 + int(50 * np.sin(t / 10)), 240), 80, (t, 180, 245 - t), -1)
                cv2.putText(frame, "FRAME SINTETICO DE TESTE", (140, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                ret, frame = cap.read()
                if not ret:
                    # Se o vídeo gravado acabar, reinicia do começo
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break

            # Processa frame no pipeline integrador
            frame_overlay, status = st.session_state.pipeline.processar_frame(frame)

            # Converte BGR -> RGB para exibição no Streamlit
            frame_rgb = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
            placeholder_video.image(frame_rgb, channels="RGB", use_column_width=True)

            # Atualiza o painel lateral com as métricas em tempo real
            frase_txt = status["frase_atual"] if status["frase_atual"] else "Aguardando término da frase..."
            glosas_list = status["buffer_glosas"]
            glosas_txt = " → ".join(glosas_list) if glosas_list else "(Buffer vazio)"

            with placeholder_frase.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">💬 Frase Traduzida em Português</div>
                    <div class="metric-value-frase">{frase_txt}</div>
                </div>
                """, unsafe_allow_html=True)

            with placeholder_glosas.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">🖐️ Glosas no Buffer Acumulado ({len(glosas_list)}/5)</div>
                    <div class="metric-value-glosas">{glosas_txt}</div>
                </div>
                """, unsafe_allow_html=True)

            with placeholder_inst.container():
                glosa_at = status["glosa_atual"]
                conf_at = status["confianca_atual"]
                st.markdown(f"**Sinal Atual Detectado:** `{glosa_at}` | **Confiança:** `{conf_at * 100:.1f}%`")

            with placeholder_topk.container():
                top_k = status.get("top_k_atual", [])
                if top_k:
                    st.caption("🔎 Palavras mais prováveis (não só a escolhida):")
                    for candidato in top_k:
                        gloss_c = candidato.get("gloss", "?")
                        conf_c = float(candidato.get("confidence", 0.0))
                        st.progress(min(max(conf_c, 0.0), 1.0), text=f"{gloss_c} — {conf_c * 100:.1f}%")

            # Pequena pausa para cadência visual de ~20 FPS no Streamlit
            time.sleep(0.05)

        if not usar_sintetico and cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
