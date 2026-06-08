import os
import csv
import json
import time
import urllib.request
import io
import re
import traceback
from groq import Groq

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
    Qualquer "N/A" soma 0.0 (logo, o teto da nota diminui naturalmente de forma justa).
    Qualquer "Não" aplica penalidade real por erro.
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
        if texto == 'N/a' or texto == 'N/A': return 'N/A'
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
                temperature=0.1
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
                # AGENTE 1: CONFORMIDADE COM NOVO RIGOR 
                # --------------------------------------------------
                print(" -> Agente 1: Analisando Conformidade e Processo com Alto Rigor...")
                prompt_agente1 = f"""
                Você é o Agente 1: Auditor Comercial Implacável. Avalie o SDR no produto {produto_detectado}.
                Sua missão é eliminar a complacência. Não dê "Sim" fácil. Seja extremamente rigoroso na análise.

                REGRAS ABSOLUTAS DE ATRIBUIÇÃO (SIM, NÃO, N/A):
                - SIM: O SDR executou a técnica com clareza OU o lead entregou o dado de bandeja espontaneamente (escuta ativa madura).
                - NÃO: O cenário existia, mas o SDR errou, foi raso, ignorou ganchos, usou diminutivos ou aceitou respostas evasivas sem contornar.
                - N/A: A oportunidade técnica NUNCA existiu na chamada (ex: lead concordou com tudo e não fez nenhuma objeção).

                MANUAL DE RIGOR ITEM A ITEM (LIGAÇÕES LONGAS > 3 MIN):
                [1. ESCUTA E ADAPTAÇÃO]
                - escuta: SIM se o SDR adaptou a conversa. NÃO se interrompeu ou ignorou falas para ler o script. N/A se a call foi 100% monólogo do lead.
                - validacao: SIM se o lead trouxe um problema e o SDR acolheu com empatia. NÃO se mudou de assunto secamente após um desabafo. N/A se o lead não expôs problemas emocionais/operacionais graves.
                - compreensao: SIM se usou dados ditos antes. NÃO se o SDR perguntou de novo algo que o cliente já tinha respondido (desatenção). N/A se a chamada caiu antes dessa validação.
                - objecoes: SIM se contornou a barreira. NÃO se o lead trouxe objeção (tempo, preço, processo) e o SDR aceitou passivamente ou desistiu. N/A apenas se o lead concordou com tudo e NÃO fez objeção alguma.

                [2. COMUNICAÇÃO E POSTURA B2B]
                - linguagem: SIM se manteve postura formal corporativa. NÃO se usou UM ÚNICO diminutivo infantilizado (sisteminha, minutinho, propostinha, tempinho) ou gíria informal. N/A se o SDR quase não falou.
                - receptividade: SIM se fez saudação acolhedora e completa. NÃO se começou ríspida, confusa ou atropelada. N/A se a gravação começou cortada.
                - rapport: SIM se quebrou o gelo usando ganchos reais (região, empresa, tom de voz). NÃO se foi robótico ou seco. N/A se o lead atendeu agressivo impossibilitando conexão.
                - discurso: SIM se usou vocabulário técnico e maduro do mercado imobiliário/ERP. NÃO se demonstrou desconhecimento ou falou bobagem técnica. N/A se abortou antes do tema principal.
                - compreensao_cliente: SIM se após uma explicação densa, checou o entendimento ("faz sentido?"). NÃO se fez monólogos gigantes sem pausar para validar se o cliente acompanhava. N/A se não houve explicação técnica.
                - clareza: SIM se fez perguntas curtas e diretas. NÃO se fez perguntas duplas, confusas ou se enrolou na dicção. N/A se o lead falou tudo sozinho.

                [3. PROCESSO E QUALIFICAÇÃO]
                - sla: 
                  * Para {produto_detectado} CRM: Coletou OBRIGATORIAMENTE Número de Corretores E Situação do CRECI. Faltou um deles, é NÃO.
                  * Para {produto_detectado} ERP: Coletou OBRIGATORIAMENTE Quantidade de Contratos E Bancos operados. Faltou um deles, é NÃO.
                  (Marque SIM se coletou ambos ou se o lead entregou de bandeja. N/A se o lead foi desqualificado precocemente).
                - spin: SIM se seguiu uma sequência exploratória de investigação. NÃO se virou panfleteiro pulando direto para as telas/recursos do sistema. N/A se o lead já despejou tudo sozinho.
                - dor: SIM se extraiu e aprofundou um gargalo real. NÃO se aceitou respostas rasas ("tá tudo bem") e mudou de assunto sem cavar o impacto financeiro/operacional. N/A se o lead foi totalmente irredutível afirmando que não tem dores.
                - gestao: SIM se mapeou quem toma a decisão final. NÃO se agendou a demo sem fazer ideia se o lead tem poder de decisão. N/A se desqualificou antes dessa etapa.
                - passos_ro: RIGOR MÁXIMO. SIM se conseguiu a confirmação VERBAL CLARA de que o lead estará na frente de um COMPUTADOR. NÃO se aceitou respostas vagas ("vou tentar", "vejo do celular", "estarei no carro") sem bater o pé e corrigir. N/A se a chamada NÃO gerou agendamento de reunião.
                - produto: SIM se conectou a solução diretamente à dor mapeada. NÃO se listou recursos genéricos sem nexo com o problema do cliente. N/A se não houve pitch de agendamento.
                - gatilhos: SIM se gerou valor e senso de compromisso com o horário. NÃO se agendou de forma desleixada ("marca qualquer hora aí"). N/A se a chamada NÃO gerou agendamento de reunião.

                REGRAS DE ERRO FATAL: Marque 'erro_fatal': true APENAS se o SDR quebrar sigilo passando preço ou agendar reunião com lead totalmente fora do perfil.
                🚨 REGRA DE JSON: NUNCA use aspas duplas (") dentro das frases de 'Evidência'. Use sempre aspas simples (').

                Retorne OBRIGATORIAMENTE o JSON preenchendo 'r' com 'Sim', 'Não' ou 'N/A':
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
                res1 = json.loads(clean_json(chat1.choices[0].message.content))
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
                
                Régua de Notas:
                - 9.0 a 10.0: Cirúrgico. Gerou urgência real e profunda usando Implicação e Necessidade maravilhosas.
                - 5.0 a 8.5: Intermediário bom. Demonstrou esforço técnico e investigou dores, mantendo a conversa fluindo de forma consultiva.
                - 0.0 a 4.5: Totalmente reativo, raso ou leu perguntas engessadas sem criar valor.

                🚨 REGRA DE FORMATAÇÃO: NUNCA use aspas duplas (") na sua justificativa. Use apenas aspas simples (').

                Responda estritamente neste formato JSON:
                {
                  "spin_scores": {"s": 5.0, "p": 6.5, "i": 4.0, "n": 3.0},
                  "analise_autoridade": "Justificativa técnica avaliando a postura do vendedor usando aspas simples."
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
                # AGENTE 3: FEEDBACK ALINHADO AO NOVO PLAYBOOK & BASE
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

                🚨 REGRA DE OURO INQUEBRÁVEL (TOLERÂNCIA ZERO PARA FEEDBACK GENÉRICO):
                É PROIBIDO usar clichês burocráticos como 'você não seguiu o playbook' ou 'faltou sequência lógica'. Se apontar um erro, FORNEÇA O TEXTO EXATO DA FALA EM FORMATO DE SCRIPT PRÁTICO.

                🚨 REGRAS CRÍTICAS DE FORMATAÇÃO JSON:
                1. Os valores das chaves DO JSON DEVEM SER STRINGS (iniciar e terminar com aspas duplas).
                2. NUNCA use aspas duplas (") DENTRO do seu texto. Use aspas simples (').
                3. NUNCA quebre a linha fisicamente. Para pular linhas no Markdown, use caracteres literais \\n.

                Estruture sua resposta estritamente com estes tópicos em Markdown usando \\n:

                ### 1. PARECER E POSTURA CONSULTIVA
                [Resumo direto de 2 linhas sobre o controle de conversa demonstrado]

                ### 2. O QUE ERROU
                - [Aponte as falhas REAIS baseadas nos 'NÃOs' apontados pelo Agente 1 com os minutos da transcrição]

                ### 3. COMO DEVERIA TER FEITO (SCRIPT PRÁTICO DA BASE DE CONHECIMENTO)
                - [Forneça a fala exata extraída das diretrizes do nosso Playbook Comercial para corrigir a falha]
                *Aviso: Consulte a aba 'Playbooks SPIN' no menu lateral.*

                ### 4. CAUSA E EFEITO NO FUNIL DE VENDAS
                - [Explique o impacto direto desse erro no esfriamento ou no no-show da reunião]

                Responda estritamente neste formato JSON:
                {
                  "parecer_executivo": "### 1. PARECER E POSTURA CONSULTIVA\\nResumo aqui.\\n\\n### 2. O QUE ERROU\\nErro aqui.\\n\\n### 3. COMO DEVERIA TER FEITO\\nCorreção baseada no Playbook aqui.\\n\\n### 4. CAUSA E EFEITO\\nEfeito aqui.",
                  "plano_de_acao_curto": "Ação exata e direta sem usar aspas duplas no meio do texto."
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
                # 4. CONSOLIDAÇÃO DA INTELIGÊNCIA MACRO (MATEMÁTICA NOVA)
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

                # Mantém retrocompatibilidade com o motor anterior do painel
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
                    "nota_geral": round(nota_geral, 1), # Chave macro de controle veloz
                    "status": status,                  # String de controle de cor (OK, ATENÇÃO, CRÍTICO)
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
                time.sleep(30)

if __name__ == "__main__":
    process_all_calls()
