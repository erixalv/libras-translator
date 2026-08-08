import json
import os

def carregar_dicionario() -> dict:
    pasta_atual = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(pasta_atual, 'dicionario.json')
    if not os.path.exists(caminho):
        return {"palavras": {}, "verbos": {}}
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)

def glosas_para_frase(glosas: list[str]) -> dict:
    if not glosas:
        return {"glosas_recebidas": [], "frase": ""}

    dados = carregar_dicionario()
    dicionario = dados.get("palavras", {})
    verbos = dados.get("verbos", {})

    frase_processada = []
    sujeito_atual = "EU"
    ultimo_verbo = None 
    foi_verbo = False
    foi_numero = False

    # Força sujeito oculto para "VOCE" (imperativo) em pedidos
    if "POR FAVOR" in [g.upper() for g in glosas] and not any(p in [g.upper() for g in glosas] for p in ["EU", "VOCE", "VOCÊ", "ELE_ELA"]):
        sujeito_atual = "VOCE"

    for glosa in glosas:
        glosa_upper = glosa.upper()

        # 1. Identificação de Pronomes (Sujeito)
        if glosa_upper in ["EU", "VOCE", "VOCÊ", "ELE_ELA"]:
            sujeito_atual = "VOCE" if glosa_upper == "VOCÊ" else glosa_upper
            frase_processada.append(dicionario.get(glosa_upper, glosa.lower()))
            foi_verbo = False
            foi_numero = False
            ultimo_verbo = None
            continue
            
        # 2. Identificação Numérica para Plural
        elif glosa_upper in ["DOIS", "TRÊS", "QUATRO", "CINCO"]:
            sujeito_atual = "ELES"
            frase_processada.append(dicionario.get(glosa_upper, glosa.lower()))
            foi_verbo = False
            foi_numero = True
            ultimo_verbo = None
            continue
            
        # 3. Forçar sujeito oculto em terceira pessoa se começar a frase
        if not frase_processada or frase_processada[-1].lower() in ["oi", "sim", "não", "por favor"]:
            if glosa_upper in ["FILHO", "VACINA", "ALUNO", "AMIGO"]:
                sujeito_atual = "ELE_ELA"

        # 4. Tratamento de Verbos
        if glosa_upper in verbos:
            if foi_verbo or ultimo_verbo == "TER": 
                frase_processada.append(glosa.lower()) # Mantém no infinitivo
                ultimo_verbo = glosa_upper # Registra o verbo mesmo no infinitivo para regras de preposição
            else:
                if sujeito_atual == "ELES":
                    # Busca plural ELES direto do JSON
                    v_conj = verbos[glosa_upper].get("ELES", glosa.lower() + "m")
                else:
                    v_conj = verbos[glosa_upper].get(sujeito_atual, glosa.lower())
                frase_processada.append(v_conj)
                ultimo_verbo = glosa_upper
            
            foi_verbo = True
            foi_numero = False
            
        # 5. Substantivos, Adjetivos e Expressões
        else:
            palavra = dicionario.get(glosa_upper, glosa.lower())
            
            # 5.1 Injeção do auxiliar "Ter" para Medo e Vontade
            if glosa_upper in ["MEDO", "VONTADE"]:
                v_ter = "têm" if sujeito_atual == "ELES" else ("tem" if sujeito_atual in ["VOCE", "ELE_ELA"] else "tenho")
                palavra = f"{v_ter} {glosa.lower()} de"
                ultimo_verbo = "TER"
                foi_verbo = True
                
            else:
                # 5.2 Conversão para plural caso a glosa anterior seja um número
                if foi_numero:
                    for artigo in ["o ", "a ", "um ", "uma "]:
                        if palavra.startswith(artigo):
                            palavra = palavra[len(artigo):] # Remove o artigo no plural
                    if not palavra.endswith("s"):
                        palavra += "s"

                # 5.3 Construção de preposições para destino e lugar
                if ultimo_verbo == "IR":
                    if glosa_upper in ["BANHEIRO", "BANCO"]:
                        palavra = "ao " + palavra.replace("o ", "")
                    elif glosa_upper == "CASA":
                        palavra = "para " + palavra
                    elif glosa_upper in ["AMERICA", "ESQUINA"]:
                        palavra = "à " + palavra.replace("a ", "")
                elif ultimo_verbo == "DORMIR" and glosa_upper == "CASA":
                    palavra = "em " + palavra

                ultimo_verbo = None
                foi_verbo = False

            frase_processada.append(palavra)
            foi_numero = False

    # 6. Montagem da frase final e correções de contração
    frase_final = " ".join(frase_processada).capitalize() + "."
    
    # Substituições gerais
    frase_final = frase_final.replace("de o ", "do ")
    frase_final = frase_final.replace("de a ", "da ")
    frase_final = frase_final.replace("de américa", "da América").replace("de América", "da América")
    frase_final = frase_final.replace("não barulho", "não faça barulho")
    frase_final = frase_final.replace("A vacina ruim", "A vacina é ruim").replace("Vacina ruim", "A vacina é ruim")
    frase_final = frase_final.replace("em a casa", "em casa")
    frase_final = frase_final.replace("américa", "América")
    frase_final = frase_final.replace("Oi o ", "Oi ")
    
    # Pronomes oblíquos
    frase_final = frase_final.replace("ajuda eu", "me ajuda")
    frase_final = frase_final.replace("ajudar eu", "me ajudar")
    frase_final = frase_final.replace("conhece eu", "me conhece")
    frase_final = frase_final.replace("vê eu", "me vê")
    
    # Concordância de gênero (Adjetivos femininos)
    frase_final = frase_final.replace("casa amarelo", "casa amarela")
    frase_final = frase_final.replace("maca amarelo", "maca amarela")
    frase_final = frase_final.replace("vacina amarelo", "vacina amarela")
    frase_final = frase_final.replace("esquina amarelo", "esquina amarela")
    
    # Correções de numerais femininos
    frase_final = frase_final.replace("dois balas", "duas balas")
    frase_final = frase_final.replace("dois macas", "duas macas")
    frase_final = frase_final.replace("dois vacinas", "duas vacinas")
    frase_final = frase_final.replace("dois esquinas", "duas esquinas")
    frase_final = frase_final.replace("dois casas", "duas casas")

    # Correções de plural em adjetivos
    frase_final = frase_final.replace("sapos amarelo", "sapos amarelos")
    frase_final = frase_final.replace("filhos amarelo", "filhos amarelos")
    frase_final = frase_final.replace("amigos amarelo", "amigos amarelos")

    # Correções de preposição pós-infinitivo e injeção de artigos perdidos
    frase_final = frase_final.replace("ir o banco", "ir ao banco")
    frase_final = frase_final.replace("ir o banheiro", "ir ao banheiro")
    frase_final = frase_final.replace("ir a esquina", "ir à esquina")
    frase_final = frase_final.replace("ir a América", "ir à América")
    frase_final = frase_final.replace("ir a casa", "ir para casa")
    frase_final = frase_final.replace("para a casa", "para casa")
    frase_final = frase_final.replace("dormir a casa", "dormir em casa")
    
    frase_final = frase_final.replace("vejo banco", "vejo o banco")
    frase_final = frase_final.replace("vejo banheiro", "vejo o banheiro")
    frase_final = frase_final.replace("veem casa", "veem a casa")
    frase_final = frase_final.replace("conhecem banco", "conhecem o banco")
    frase_final = frase_final.replace("conhecer banco", "conhecer o banco")
    frase_final = frase_final.replace("conhece banco", "conhece o banco")
    frase_final = frase_final.replace("aproveita banheiro", "aproveita o banheiro")
    frase_final = frase_final.replace("aproveito banco", "aproveito o banco")
    frase_final = frase_final.replace("gosta de banco", "gosta do banco")
    
    # Remove espaços duplos remanescentes
    frase_final = " ".join(frase_final.split())

    return {
        "glosas_recebidas": glosas,
        "frase": frase_final
    }

if __name__ == "__main__":
    caminho_testes = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teste.json')
    
    with open(caminho_testes, 'r', encoding='utf-8') as f:
        testes = json.load(f)

    for i, teste in enumerate(testes, 1):
        resultado = glosas_para_frase(teste)
        print(f"Teste {i}: {resultado['frase']}")