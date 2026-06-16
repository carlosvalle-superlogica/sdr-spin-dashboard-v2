import os
import csv
import json
import time
import urllib.request
import io
import re
import traceback
from groq import Groq
from pydantic import BaseModel, Field
from typing import Literal

# ==========================================
# 0. BLINDAGEM DE DADOS COM PYDANTIC (NÍVEL 11/10)
# ==========================================
# Aqui nós obrigamos fisicamente o modelo a responder APENAS com Sim, Não ou N/A.
class AvaliacaoItem(BaseModel):
    r: Literal["Sim", "Não", "N/A"] = Field(description="OBRIGATÓRIO: Apenas 'Sim', 'Não' ou 'N/A' baseado nas regras de silêncio e proatividade.")
    e: str = Field(description="Evidência real extraída da transcrição com aspas simples.")

class OperacionalAuditoria(BaseModel):
    escuta: AvaliacaoItem
    validacao: AvaliacaoItem
    compreensao: AvaliacaoItem
    objecoes: AvaliacaoItem
    linguagem: AvaliacaoItem
    receptividade: AvaliacaoItem
    rapport: AvaliacaoItem
    discurso: AvaliacaoItem
    compreensao_cliente: AvaliacaoItem
    clareza: AvaliacaoItem
    sla: AvaliacaoItem
    spin: AvaliacaoItem
    dor: AvaliacaoItem
    gestao: AvaliacaoItem
    passos_ro: AvaliacaoItem
    produto: AvaliacaoItem
    gatilhos: AvaliacaoItem

class AuditoriaAgente1(BaseModel):
    erro_fatal: bool = Field(description="True APENAS se quebrou sigilo de preço ou agendou lead fora do perfil.")
    operacional: OperacionalAuditoria

# ==========================================
# 1. CONFIGURAÇÕES E VARIÁVEIS DE AMBIENTE
# ==========================================
GROQ_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_KEY:
    raise ValueError("ERRO CRÍTICO: GROQ_API_KEY não encontrada nos Secrets!")

client = Groq(api_key=GROQ_KEY)
CSV_FILE = "dados_chamadas.csv"
CONSOLIDATED_FILE = "consolidated_data.json"
PORTAL_ID = "20131994"

# Modelos ativos oficiais estáveis da API Groq
MODELO_RAPIDO = "llama-3.1-8b-instant"
MODELO_PARERES = "llama-3.3-70b-versatile"

# ==========================================
# 2. SISTEMAS DE SEGURANÇA E MATEMÁTICA RECALIBRADA
# ==========================================
def clean_json(text):
    """Garante a limpeza e extração apenas do objeto JSON retornado pelas APIs de LLM."""
    text = text.strip()
    
    if text.startswith("```json"): 
        text = text[7:]
    elif text.startswith("```"): 
        text = text[3:]
    
    if text.endswith("```"): 
        text = text[:-3]
    
    text = text.strip()
    
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: 
            return match.group(0)
    except Exception: 
        pass
    
    return text

def safe_float(val, default=0.0):
    try: 
        return float(val)
    except Exception: 
        return default

def calcular_segundos(duracao_str):
    """Converte strings de duração formatadas (HH:mm:ss ou mm:ss) em segundos totais."""
    try:
        partes = duracao_str.split(':')
        if len(partes) == 3: 
            return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])
        if len(partes) == 2: 
            return int(partes[0]) * 60 + int(partes[1])
    except Exception: 
        pass
    
    return 1

def calcular_nota_operacional(op_data, erro_fatal):
    """
    MATEMÁTICA ADITIVA RÍGIDA (A BUSCA PELO 10.0):
    O SDR começa com 0.0. Para tirar 10.0, precisa de 17 "Sim".
    Qualquer "N/A" soma 0.0 (logo, o teto da nota diminui naturalmente de forma justa, pois foi uma ligação mais fácil).
    Qualquer "Não" aplica penalidade real por erro ou oportunidade desperdiçada.
    """
    nota = 0.0

    chaves_criticas = ['sla', 'passos_ro', 'gestao']  
    # 3 itens (1.0 cada = 3.0 max)
    
    chaves_estrategicas = ['spin', 'dor', 'validacao', 'objecoes', 'produto', 'escuta', 'compreensao'] 
    # 7 itens (0.7 cada = 4.9 max)
    
    chaves_formais = ['linguagem', 'receptividade', 'rapport', 'discurso', 'compreensao_cliente', 'clareza', 'gatilhos'] 
    # 7 itens (0.3 cada = 2.1 max)

    # Função interna para limpar o texto que a IA devolve (Garante precisão no cálculo)
    def normalizar_resposta(valor):
        texto = str(valor).strip().title()
        if texto == 'Nao': return 'Não'
        if texto.upper() == 'N/A' or texto == 'N/a': return 'N/A'
        return texto

    # --- TIER 1: CRÍTICOS ---
    for k in chaves_criticas:
        r = normalizar_resposta(op_data.get(k, {}).get('r', ''))
        if r == 'Sim': 
            nota += 1.0
        elif r == 'Não': 
            nota -= 1.0 # Penalidade grave

    # --- TIER 2: ESTRATÉGICOS ---
    for k in chaves_estrategicas:
        r = normalizar_resposta(op_data.get(k, {}).get('r', ''))
        if r == 'Sim': 
            nota += 0.7
        elif r == 'Não': 
            nota -= 0.5 # Penalidade média

    # --- TIER 3: FORMAIS ---
    for k in chaves_formais:
        r = normalizar_resposta(op_data.get(k, {}).get('r', ''))
        if r == 'Sim': 
            nota += 0.3
        elif r == 'Não': 
            nota -= 0.2 # Penalidade leve

    # --- ERRO FATAL ---
    if erro_fatal:
        nota -= 4.0

    # CLAMPEAMENTO SEGURO
    return min(max(nota, 0.0), 10.0)

def executar_chat_com_retentativa(model, messages, response_format, max_retries=6):
    """Executa chamadas à API do Groq controlando de forma inteligente erros de Rate Limit (429)."""
    base_delay = 15  
    
    for attempt in range(max_retries):
        try:
            chat = client.chat.completions.create(
                model=model, 
                messages=messages, 
                response_format=response_format, 
                temperature=0.0 # CRÍTICO: Zero criatividade. O modelo deve ser determinístico na auditoria.
            )
            return chat
            
        except Exception as e:
            err_msg = str(e).lower()
            
            # Captura qualquer erro de limite de requisição ou 429
            if "429" in err_msg or "rate" in err_msg or "too many" in err_msg:
                match = re.search(r"try again in ([0-9.]+)(s|ms)", err_msg)
                
                if match:
                    wait_time = float(match.group(1))
                    if match.group(2) == "ms": 
                        wait_time = wait_time / 1000.0
                else:
                    wait_time = base_delay * (attempt + 1)
                
                wait_time += 5.0 # Margem de segurança
                print(f"   ⚠️ [RATE LIMIT] Limite da API atingido. Aguardando {wait_time:.1f}s (Tentativa {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                
            else:
                raise e
                
    raise RuntimeError(f"Erro: Falha persistente na API da Groq após {max_retries} tentativas.")

# ==========================================
# 3. PIPELINE DE EXECUÇÃO MULTIAGENTE
# ==========================================
def process_all_calls():
    
    if not os.path.exists(CSV_FILE):
        print(f"Erro: Ficheiro {CSV_FILE} não encontrado.")
        return
        
    db = {}
    
    if os.path.exists(CONSOLIDATED_FILE):
        try:
            with open(CONSOLIDATED_FILE, 'r', encoding='utf-8') as f: 
                db = json.load(f)
        except Exception: 
            db = {}

    with open(CSV_FILE, mode='r', encoding='utf-8-sig') as f:
        sample = f.read(2048)
        delimiter = ';' if ';' in sample else ','
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            call_id = row.get("ID do objeto", "").strip()
            audio_url = row.get("URL de gravação", "").strip()
            result = row.get("Resultado da chamada", "").strip()
            sdr_name = row.get("Atividade atribuída a", "").strip() or "SDR"
            date_str = row.get("Data da atividade", "").strip()
            duration = row.get("Duração da chamada (HH:mm:ss)", "").strip() or "00:00"
            title = row.get("Título da chamada", "").strip()
            
            # Recupera IDs associados e constrói dinamicamente a URL do HubSpot
            deal_id = row.get("Associated Deal IDs", "").strip()
            deal_url = ""
            if deal_id:
                primeiro_id = deal_id.split(',')[0].strip()
                deal_url = f"[https://app.hubspot.com/contacts/](https://app.hubspot.com/contacts/){PORTAL_ID}/deal/{primeiro_id}/"

            if not call_id or not audio_url.startswith("http") or result.lower() not in ["ligação atendida", "connected", "atendida"] or call_id in db:
                continue

            print(f"\n=======================================================")
            print(f"🔥 INICIANDO AUDITORIA | ID: {call_id} | SDR: {sdr_name}")
            print(f"=======================================================")
            
            txt_verif = (title + " " + json.dumps(row)).lower()
            produto_detectado = "CRM" if any(p in txt_verif for p in ["crm", "creci", "corretor"]) else "ERP"

            # Trava de Segurança Aprimorada para Download de Áudio (Timeout de 30s)
            try:
                req = urllib.request.Request(audio_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as response: 
                    audio_bytes = response.read()
            except Exception as e:
                print(f"   ⚠️ [TIMEOUT/ERRO DOWNLOAD] Servidor de áudio falhou ou demorou muito: {e}. Pulando...")
                continue

            try:
                # Prevenção Ativa Contra Loop de Arquivos Enormes na Groq: Tolerância máxima de 20MB.
                tamanho_mb = len(audio_bytes) / (1024 * 1024)
                if tamanho_mb > 20.0:
                    print(f"   ⚠️ [PULANDO CHAMADA] O arquivo possui {tamanho_mb:.2f} MB excedendo o teto seguro de 20MB da API.")
                    continue

                # Transcrição com Whisper-Large-V3
                transcription = client.audio.transcriptions.create(
                    file=("audio.mp3", io.BytesIO(audio_bytes)), 
                    model="whisper-large-v3", 
                    response_format="json"
                )
                
                texto = transcription.text
                
                if len(texto) < 10: 
                    print("Chamada ignorada: Áudio sem conteúdo legível ou muito curto.")
                    continue

                segundos = calcular_segundos(duration)
                wps = round(len(texto.split()) / segundos, 2) if segundos > 0 else 0.0

                # --------------------------------------------------
                # AGENTE 1: CONFORMIDADE COM N/A ESTRATÉGICO
                # --------------------------------------------------
                print(" -> Agente 1: Analisando Conformidade e Adaptação com Alto Rigor...")
                prompt_agente1 = f"""
                Você é o Agente 1: Auditor Comercial Implacável. Avalie o SDR no produto {produto_detectado}.
                Sua missão é eliminar a complacência. Não dê "Sim" fácil. Seja extremamente rigoroso na análise.

                MUITO IMPORTANTE - USO ESTRATÉGICO DO N/A (A REGRA DE OURO):
                O 'N/A' (Não Aplicável) SÓ DEVE SER CONSIDERADO em duas situações exclusivas:
                1) Quando NÃO houver a necessidade do SDR fazer a pergunta porque o Lead mesmo já informou antes do SDR precisar perguntar (A proatividade do cliente anula a necessidade da técnica).
                2) Quando o cenário da técnica NUNCA existiu na ligação (Ex: o lead não fez nenhuma objeção, logo não houve o que contornar).
                
                É ESTREITAMENTE PROIBIDO usar N/A se o SDR falhou, ignorou um gancho ou cometeu um erro. Nesses casos, a nota é NÃO.

                DIRETRIZES DE AUDITORIA ITEM A ITEM:
                [1. ESCUTA E ADAPTAÇÃO]
                - escuta: O SDR adaptou a conversa? Se interrompeu o lead ou ignorou uma fala, marque 'Não'.
                - validacao: Marque 'N/A' APENAS se o lead não expôs nenhum problema. Se expôs e o SDR acolheu, marque 'Sim'. Se mudou de assunto, marque 'Não'.
                - compreensao: Inteligência de fluxo. Marque 'Não' se o SDR perguntou de novo algo que o lead já tinha respondido antes.
                - objecoes: Contornou barreiras? Se o lead não apresentou nenhuma objeção, marque OBRIGATORIAMENTE 'N/A'. Se apresentou e o SDR falhou, marque 'Não'.

                [2. COMUNICAÇÃO E POSTURA B2B]
                - linguagem: Norma culta. ATENÇÃO: Se o SDR usou um único diminutivo (sisteminha, minutinho, propostinha), marque 'Não'.
                - receptividade: Executou a saudação completa de forma acolhedora? Marque 'Não' se começou ríspido.
                - rapport: Quebrou o gelo? Marque 'Não' se iniciou um interrogatório seco. Marque 'N/A' se o lead atendeu agressivo matando o rapport.
                - discurso: Usou vocabulário técnico correto do mercado imobiliário/ERP?
                - compreensao_cliente: Validou com o cliente se ele entendeu a explicação técnica?
                - clareza: Fez perguntas curtas e diretas? Marque 'Não' se fez perguntas duplas ou confusas.

                [3. PROCESSO E QUALIFICAÇÃO]
                - sla: 
                  * Para {produto_detectado} CRM: Coletou Número de Corretores E Situação do CRECI?
                  * Para {produto_detectado} ERP: Coletou Quantidade de Contratos E Bancos operados?
                  (Marque 'N/A' APENAS se o lead entregou de bandeja e o SDR não precisou perguntar. Se faltou coletar 1 item, marque 'Não').
                - spin: Seguiu a sequência exploratória ou só apresentou o sistema igual um panfleto?
                - dor: Encontrou um gargalo real? Se o lead deu respostas vazias e o SDR não insistiu, marque 'Não'. Marque 'N/A' APENAS se o lead listou todas as dores sem o SDR perguntar.
                - gestao: Mapeou quem toma a decisão final? Marque 'N/A' APENAS se o lead avisou espontaneamente que é o dono/decisor.
                - passos_ro: RIGOR MÁXIMO. Conseguiu a confirmação VERBAL CLARA de que o lead estará num COMPUTADOR na próxima reunião? Se aceitou "ver pelo celular/carro", marque 'Não'. Se a call não gerou agenda, marque 'N/A'.
                - produto: Conectou a solução tecnológica à dor de forma inteligente?
                - gatilhos: Gerou urgência de agenda? Marque 'N/A' se não teve agendamento.

                REGRAS DE ERRO FATAL E JSON: 
                - Marque 'erro_fatal': true APENAS se o SDR quebrar o sigilo e passar preço ou agendar reunião com lead fora de perfil.
                - 🚨 NUNCA use aspas duplas (") dentro das suas frases de 'Evidência'. Use sempre aspas simples (').

                Retorne OBRIGATORIAMENTE o JSON preenchendo 'r' estritamente com 'Sim', 'Não' ou 'N/A':
                {{
                  "erro_fatal": false,
                  "operacional": {{
                    "escuta": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "validacao": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "compreensao": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "objecoes": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "linguagem": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "receptividade": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "rapport": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "discurso": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "compreensao_cliente": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "clareza": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "sla": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "spin": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "dor": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "gestao": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "passos_ro": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}, 
                    "produto": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}},
                    "gatilhos": {{"r": "[Sim/Não/N/A]", "e": "Evidencia real com aspas simples"}}
                  }}
                }}
                """
                chat1 = executar_chat_com_retentativa(
                    model=MODELO_RAPIDO, 
                    messages=[{"role": "system", "content": prompt_agente1}, {"role": "user", "content": texto}], 
                    response_format={"type": "json_object"}
                )
                
                # VALIDAÇÃO PYDANTIC
                res1_bruto = json.loads(clean_json(chat1.choices[0].message.content))
                AuditoriaAgente1(**res1_bruto) # Blindagem: Se o JSON não tiver estritamente Sim, Não ou N/A, vai gerar erro e forçar retentativa
                res1 = res1_bruto
                time.sleep(2)

                # --------------------------------------------------
                # AGENTE 2: SPIN SCORE
                # --------------------------------------------------
                print(" -> Agente 2: Avaliando Notas de Metodologia SPIN...")
                prompt_agente2 = """
                Você é o Agente 2: Especialista em Metodologia SPIN e Psicologia Comercial.
                Avalie o nível de aprofundamento das perguntas realizadas pelo SDR.
                - S (Situação): Mapeamento do cenário atual.
                - P (Problema): Investigação dos gargalos e dores.
                - I (Implicação): Investigação das consequências de não resolver o problema (gera urgência). Rigor extremo aqui.
                - N (Necessidade de Solução): Fez o cliente declarar o valor da solução.
                
                INSTRUÇÕES DE NOTAS:
                - Notas 9.0 a 10.0: Seja extremamente rigoroso. Só dê nota máxima se o SDR foi cirúrgico, tocou na ferida e gerou urgência inquestionável.
                - Notas 5.0 a 8.5: Intermediário bom. O SDR tentou investigar e manteve a conversa fluindo de forma consultiva.
                - Notas 0.0 a 4.5: O SDR foi totalmente reativo, raso ou leu perguntas engessadas sem criar valor.

                🚨 REGRA DE FORMATAÇÃO: NUNCA use aspas duplas (") na sua justificativa, pois quebra o JSON. Use apenas aspas simples (').

                Responda estritamente neste formato JSON:
                {
                  "spin_scores": {"s": 5.0, "p": 6.5, "i": 4.0, "n": 3.0},
                  "analise_autoridade": "Breve justificativa técnica avaliando a postura do vendedor usando aspas simples."
                }
                """
                chat2 = executar_chat_com_retentativa(
                    model=MODELO_RAPIDO, 
                    messages=[{"role": "system", "content": prompt_agente2}, {"role": "user", "content": texto}], 
                    response_format={"type": "json_object"}
                )
                res2 = json.loads(clean_json(chat2.choices[0].message.content))
                
                # 🚨 RESPIRO ABSOLUTO DE 35 SEGUNDOS PARA ZERAR O RATE LIMIT DO MODELO 70B 🚨
                print("   ⏳ Dando fôlego estratégico (35s) para a cota da IA limpar antes do modelo de pareceres...")
                time.sleep(35)

                # --------------------------------------------------
                # AGENTE 3: FEEDBACK ALINHADO AO NOVO PLAYBOOK
                # --------------------------------------------------
                print(" -> Agente 3: Construindo Feedback Técnico Alinhado com o Novo Playbook...")
                contexto_sintese = f"Resultados Agente 1: {json.dumps(res1)}\nResultados Agente 2: {json.dumps(res2)}"
                prompt_agente3 = """
                Você é o Diretor de Enablement. Sua missão é dar feedback de alta performance totalmente alinhado com o nosso Playbook e Base de Conhecimento Rígida.
                Você deve ser o treinador de elite. Se o Agente 1 apontou uma falha (NÃO), você deve cruzar com o Playbook e ensinar como reverter.

                🚨 DIRETRIZ DA BASE DE CONHECIMENTO E PLAYBOOKS COMERCIAIS:
                - Se falhou em 'passos_ro' (aceitou celular/carro): Ensine o script de barreira de tela. Ex: 'Em vez de aceitar, use o Playbook: Perfeito, fulano, mas como vou te mostrar as telas de contratos e conciliação de bancos, preciso que você esteja em telas grandes para avaliar 100%. Conseguimos ajustar o horário para quando você estiver no escritório?'
                - Se falhou em 'dor' (aceitou resposta rasa): Ensine a técnica de desdobramento de impacto financeiro. Ex: 'Quando o cliente disser que o sistema atual é lento, não mude de assunto. Pergunte: E hoje, quanto tempo a sua equipe perde refazendo esse processo na mão por causa dessa lentidão?'
                - Se falhou em 'linguagem' (usou diminutivo): Alerte sobre a quebra de postura corporativa sênior B2B.

                🚨 REGRA DE OURO INQUEBRÁVEL (TOLERÂNCIA ZERO PARA FEEDBACK GENÉRICO E PALESTRAS DE IA):
                É EXPRESSAMENTE PROIBIDO usar palavras vazias e burocráticas como 'você não seguiu o playbook'. 
                Se você apontar um erro, VOCÊ DEVE OBRIGATORIAMENTE FORNECER A FALA EXATA que o vendedor deveria ter usado no lugar.

                🚨 REGRAS CRÍTICAS DE FORMATAÇÃO JSON (ANTI-ERRO):
                1. Os valores das chaves DO JSON DEVEM SER STRINGS (iniciar e terminar com aspas duplas).
                2. NUNCA use aspas duplas (") DENTRO do seu texto. Se precisar citar algo, use aspas simples (').
                3. NUNCA quebre a linha fisicamente. Para pular linhas e formatar os tópicos em Markdown, use OBRIGATORIAMENTE os caracteres literais \\n.

                Estruture SUA resposta OBRIGATORIAMENTE com estes tópicos em Markdown usando \\n:

                ### 1. PARECER E POSTURA CONSULTIVA
                [Um resumo direto de 2 linhas sobre o controle de conversa demonstrado]

                ### 2. O QUE ERROU
                - [Aponte falhas REAIS encontradas na transcrição baseadas nos NÃOs]

                ### 3. COMO DEVERIA TER FEITO (SCRIPT PRÁTICO DA BASE DE CONHECIMENTO)
                - [Forneça o texto exato em formato de fala]

                ### 4. CAUSA E EFEITO NO FUNIL DE VENDAS
                - [Explique de forma direta como esse erro esfria o lead.]

                Responda estritamente neste formato JSON:
                {
                  "parecer_executivo": "### 1. PARECER E POSTURA CONSULTIVA\\nResumo aqui.\\n\\n### 2. O QUE ERROU\\nErro aqui.\\n\\n### 3. COMO DEVERIA TER FEITO\\nCorreção aqui.\\n\\n### 4. CAUSA E EFEITO\\nEfeito aqui.",
                  "plano_de_acao_curto": "Ação exata e direta sem aspas duplas internas."
                }
                """
                
                chat3 = executar_chat_com_retentativa(
                    model=MODELO_PARERES, 
                    messages=[
                        {"role": "system", "content": prompt_agente3}, 
                        {"role": "user", "content": f"Contexto Analítico: {contexto_sintese}\nTranscrição da Chamada: {texto}"}
                    ], 
                    response_format={"type": "json_object"}
                )
                res3 = json.loads(clean_json(chat3.choices[0].message.content))

                # --------------------------------------------------
                # 4. CONSOLIDAÇÃO DA INTELIGÊNCIA MACRO E SEMÁFORO
                # --------------------------------------------------
                s_spin = res2.get("spin_scores", {})
                nota_spin = sum([safe_float(s_spin.get(k)) for k in ['s','p','i','n']]) / 4.0
                nota_op = calcular_nota_operacional(res1.get("operacional", {}), res1.get("erro_fatal", False))
                
                # FÓRMULA PONDERADA DOS PARAMETROS MACRO (60% Conformidade / 40% SPIN)
                nota_geral = (nota_op * 0.6) + (nota_spin * 0.4)
                
                # MOTOR DE STATUS SEGURO (SEMÁFORO DE PERFORMANCE)
                if res1.get("erro_fatal", False) or nota_geral <= 6.4:
                    status = "CRÍTICO"   # Vermelho 🔴
                elif nota_geral <= 8.4:
                    status = "ATENÇÃO"   # Amarelo 🟡
                else:
                    status = "OK"        # Verde 🟢

                # Mantém retrocompatibilidade com o painel antigo
                urgencia = "SIM" if status == "CRÍTICO" else "NÃO"

                db[call_id] = {
                    "id": call_id, 
                    "sdr": sdr_name, 
                    "produto": produto_detectado, 
                    "data": date_str, 
                    "duracao": duration,
                    "wps": wps, 
                    "nota_spin": round(nota_spin, 1), 
                    "nota_op": round(nota_op, 1),
                    "nota_geral": round(nota_geral, 1), 
                    "status": status,                  
                    "urgencia": urgencia, 
                    "deal_url": deal_url, 
                    "audio_url": audio_url,
                    "notas_s_p_i_n": s_spin, 
                    "formulario": res1.get("operacional", {}),
                    "parecer": res3.get("parecer_executivo", ""), 
                    "sugestoes": res3.get("plano_de_acao_curto", ""),
                    "transcricao": texto
                }
                
                with open(CONSOLIDATED_FILE, 'w', encoding='utf-8') as sf: 
                    json.dump(db, sf, ensure_ascii=False, indent=4)
                
                print(f"✅ Auditoria Finalizada com Sucesso! GERAL: {nota_geral:.1f} ({status}) | SPIN: {nota_spin:.1f} | Op: {nota_op:.1f}")
                
                # Zera o fluxo final de requisições com uma folga antes da próxima linha do CSV
                time.sleep(10)

            except Exception as e:
                print(f"❌ Erro na auditoria do ID {call_id}: {e}")
                traceback.print_exc()
                # Em caso de erro pesado, o robô dorme e recupera as forças
                time.sleep(30)

if __name__ == "__main__":
    process_all_calls()
