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

# Modelos ativos oficiais estáveis da API Groq em 2026
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
    except: 
        pass
    return text

def safe_float(val, default=0.0):
    try: 
        return float(val)
    except: 
        return default

def calcular_segundos(duracao_str):
    """Converte strings de duração formatadas (HH:mm:ss ou mm:ss) em segundos totais."""
    try:
        partes = duracao_str.split(':')
        if len(partes) == 3: 
            return int(partes[0])*3600 + int(partes[1])*60 + int(partes[2])
        if len(partes) == 2: 
            return int(partes[0])*60 + int(partes[1])
    except: 
        pass
    return 1

def calcular_nota_operacional(op_data, erro_fatal):
    """
    MATEMÁTICA DE AUDITORIA DE TOLERÂNCIA ZERO:
    - O SDR inicia com a nota máxima de 10.0 (Exigência de perfeição).
    - O 'Sim' mantém a nota estável.
    - O 'Não' indica quebra ativa de processo e penaliza agressivamente (-1.5 pontos).
    - O 'N/A' indica omissão e penaliza (-0.5 pontos), EXCETO para 'objecoes' 
      (pois se o cliente não fez objeção, o SDR não pode ser punido por isso).
    """
    if erro_fatal: 
        return 0.0

    # Consolidação estrita de todas as 17 chaves de checagem
    todas_chaves = [
        'escuta', 'validacao', 'compreensao', 'objecoes',
        'linguagem', 'receptividade', 'rapport', 'discurso', 'compreensao_cliente', 'clareza',
        'sla', 'spin', 'dor', 'gestao', 'passos_ro', 'produto', 'gatilhos'
    ]

    # Contagem exata das falhas cometidas pelo SDR ao longo da ligação
    total_nao = sum(1 for k in todas_chaves if op_data.get(k, {}).get('r') == 'Não')
    
    # Conta N/A para todas as chaves, exceto 'objecoes' para não punir chamadas perfeitamente lisas
    total_na = sum(1 for k in todas_chaves if k != 'objecoes' and op_data.get(k, {}).get('r') == 'N/A')

    # Definição dos pesos de punição
    PENALIDADE_NAO = 1.5  # Erro grave / Quebra de política comercial
    PENALIDADE_NA = 0.5   # Omissão técnica / Deixou de colher informação

    # Cálculo dedutivo
    nota_final = 10.0 - (total_nao * PENALIDADE_NAO) - (total_na * PENALIDADE_NA)
    
    # Clampa rigidamente o resultado entre o piso de 0.0 e o teto de 10.0
    return min(max(nota_final, 0.0), 10.0)

def executar_chat_com_retentativa(model, messages, response_format, max_retries=5):
    """Executa chamadas à API do Groq controlando de forma inteligente erros de Rate Limit (429)."""
    base_delay = 5  
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
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                match = re.search(r"try again in ([0-9.]+)(s|ms)", err_msg)
                wait_time = float(match.group(1)) if match else (base_delay ** (attempt + 1))
                if match and match.group(2) == "ms":
                    wait_time = wait_time / 1000.0
                
                wait_time = max(wait_time + 1.0, 3.0) 
                print(f"   ⚠️ [RATE LIMIT] Limite atingido. Aguardando {wait_time}s antes da tentativa {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Erro: Falha persistente por excesso de requisições no Groq (Rate Limit).")

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
        except: 
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
            print(f"🔥 INICIANDO AUDITORIA MULTIAGENTE | ID: {call_id} | SDR: {sdr_name}")
            print(f"=======================================================")
            
            txt_verif = (title + " " + json.dumps(row)).lower()
            produto_detectado = "CRM" if any(p in txt_verif for p in ["crm", "creci", "corretor"]) else "ERP"

            # Trava de Segurança Isolada para Download de Áudio (Timeout de 15 segundos)
            try:
                req = urllib.request.Request(audio_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as response: 
                    audio_bytes = response.read()
            except Exception as e:
                print(f"   ⚠️ [TIMEOUT/ERRO DOWNLOAD] Servidor de áudio falhou ou demorou muito: {e}. Pulando...")
                continue

            try:
                # Prevenção Ativa Contra Erro 413: Mede o tamanho em MB antes de enviar para a API
                tamanho_mb = len(audio_bytes) / (1024 * 1024)
                if tamanho_mb > 25.0:
                    print(f"   ⚠️ [PULANDO CHAMADA] O arquivo possui {tamanho_mb:.2f} MB excedendo o teto de 25MB da API.")
                    continue

                # 2. Transcrição com Whisper-Large-V3
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
                # AGENTE 1: O Auditor Processual Rígido (Conformidade)
                # --------------------------------------------------
                print(" -> Executando Agente 1: Auditoria de Processos...")
                prompt_agente1 = f"""
                Você é o Agente 1: Auditor de Processos Inflexível e Rígido. Sua missão é analisar a transcrição da chamada e julgar o cumprimento exato das políticas comerciais e de qualificação da empresa para o produto {produto_detectado}.
                Seja agressivo, extremamente criterioso e intolerante com falhas. Menções vagas, superficiais ou desculpas aceitas passivamente pelo SDR DEVEM receber a marcação "Não".

                DIRETRIZES DE JULGAMENTO CIRÚRGICO (GABARITO DE CONFORMIDADE):
                
                [1. ESCUTA E ADAPTAÇÃO]
                - escuta: Marque "Sim" apenas se o SDR ouviu as dores sem cortar o raciocínio do cliente. Se o SDR atropelou ou ignorou uma fala importante, marque "Não".
                - validacao: Marque "Sim" se o SDR ancorou o problema do cliente antes de seguir em frente (ex: "Entendi, então o seu principal gargalo hoje é o controle manual, correto?"). Se ele agiu como um mero leitor de questionário, marque "Não".
                - compreensao: O SDR entendeu o contexto de primeira. Se ele repetiu uma pergunta que o lead já tinha respondido antes por falta de atenção, marque "Não".
                - objecoes: Como o SDR lidou com barreiras? Se o lead trouxe uma objeção (tempo, dinheiro, sistema atual) e o SDR contornou com argumento técnico, marque "Sim". Se o lead aceitou passivamente a barreira e recuou, marque "Não". Marque "N/A" apenas se a conversa correu 100% lisa sem nenhuma objeção do lead.

                [2. COMUNICAÇÃO]
                - linguagem: Uso de vocabulário corporativo limpo. Se o SDR abusou de gírias ou vícios de linguagem excessivos e irritantes (tipo, né, tá, aham, saca), marque "Não".
                - receptividade: Tom de voz cordial, consultivo e enérgico. Se demonstrou desânimo, pressa ou tédio, marque "Não".
                - rapport: Conexão real. Se o SDR usou uma abordagem empática inicial baseada no que o lead falou, marque "Sim". Se pareceu um robô de telemarketing operando um script engessado, marque "Não".
                - discurso: Demonstrou autoridade. O SDR se posicionou como um especialista que entende do mercado do cliente?
                - compreensao_cliente: O cliente entendeu as explicações do SDR de primeira, sem demonstrar confusão ou pedir para reexplicar?
                - clareza: Perguntas diretas e limpas. Se o SDR fez perguntas duplas, confusas ou prolixas que o cliente demorou para entender, marque "Não".

                [3. PROCESSO E POLÍTICAS DE QUALIFICAÇÃO]
                - sla: INVESTIGAÇÃO COMPLETA DOS DADOS DE SLA. 
                  * Para {produto_detectado} CRM: O SDR OBRIGATORIAMENTE precisa extrair o Número de Corretores E a existência/situação do CRECI da imobiliária.
                  * Para {produto_detectado} ERP: O SDR OBRIGATORIAMENTE precisa extrair a Quantidade de Contratos Ativos E os Bancos com os quais o cliente opera.
                  Se o SDR esqueceu ou deixou de perguntar QUALQUER um desses dois dados específicos do produto correspondente, marque "Não".
                - spin: Investigação profunda. O SDR explorou o problema do cliente fazendo perguntas de causa ou o SDR fez um monólogo falando apenas da nossa empresa?
                - dor: Identificação da raiz do problema. O lead confessou um impacto ou gargalo real na operação? (Se o SDR aceitou um "está tudo bem, só quero conhecer", marque "Não").
                - gestao: Mapeamento de Decisão. O SDR validou quem toma a decisão final ou se existem outros sócios/diretores envolvidos no processo?
                - passos_ro (COMPROMISSO ESTREITO DE COMPUTADOR): Esta é a nossa política de demonstração mais crítica. O SDR precisa firmar o compromisso explícito de que o lead estará na frente de um COMPUTADOR (Desktop/Laptop). Se o lead disse "vou tentar", "acho que sim", "vou estar dirigindo", "posso ver pelo celular" ou se o SDR apenas fez uma menção morna e aceitou uma resposta incerta, marque "Não". Exija confirmação verbal clara do lead de estar na frente de um computador.
                - produto: O SDR apresentou um gancho de valor personalizado conectando o {produto_detectado} diretamente à dor que o lead acabou de confessar?
                - gatilhos: O SDR gerou senso de urgência ou escassez de agenda para valorizar o horário com o especialista?

                REGRAS DE VIOLAÇÃO DE POLÍTICA (ERRO FATAL):
                Você DEVE marcar "erro_fatal": true se o SDR cometer qualquer uma das infrações abaixo:
                1. PREÇO: Se o SDR violou a política de sigilo de valores e passou preços, tabelas, estimativas financeiras, taxas ou valores mensais (ex: "custa a partir de R$...", "fica na faixa de X").
                2. AGENDAMENTO DE LEAD INVÁLIDO: Se o SDR aceitou agendar a reunião com um lead que NÃO possui os critérios mínimos do produto (Para CRM: lead com 0 corretores ou sem CRECI definitivo/em andamento. Para ERP: lead com 0 contratos ativos).

                Retorne OBRIGATORIAMENTE este formato JSON preenchendo o campo "r" APENAS com "Sim", "Não" ou "N/A":
                {{
                  "erro_fatal": false,
                  "operacional": {{
                    "escuta": {{"r": "[Sim/Não/N/A]", "e": "Frase exata extraída da transcrição que comprova o seu julgamento rigoroso"}},
                    "validacao": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "compreensao": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "objecoes": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "linguagem": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "receptividade": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "rapport": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "discurso": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "compreensao_cliente": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "clareza": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "sla": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "spin": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "dor": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "gestao": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "passos_ro": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "produto": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}},
                    "gatilhos": {{"r": "[Sim/Não/N/A]", "e": "Evidência"}}
                  }}
                }}
                """
                chat1 = executar_chat_com_retentativa(
                    model=MODELO_RAPIDO,
                    messages=[{"role": "system", "content": prompt_agente1}, {"role": "user", "content": texto}],
                    response_format={"type": "json_object"}
                )
                res1 = json.loads(clean_json(chat1.choices[0].message.content))
                time.sleep(4)

                # --------------------------------------------------
                # AGENTE 2: O Cientista de Método e SPIN Selling (Aderência)
                # --------------------------------------------------
                print(" -> Executando Agente 2: Avaliação de Metodologia Comercial...")
                prompt_agente2 = """
                Você é o Agente 2: Especialista em SPIN Selling e Psicologia de Vendas. Avalie a profundidade técnica do discovery do SDR.
                Analise se o SDR tomou as rédeas da conversa ou se foi reativo. Atribua notas de 0.0 a 10.0 para cada pilar do SPIN.
                Abaixe a nota drasticamente se o SDR aceitou respostas prontas sem fazer perguntas investigativas de follow-up.
                Exija que a etapa de Implicação (I) tenha gerado desconforto ou mensuração de perdas no lead.

                Responda estritamente neste formato JSON:
                {{
                  "spin_scores": {{"s": 5.0, "p": 4.5, "i": 1.0, "n": 2.0}},
                  "analise_autoridade": "Descreva o controle de conversa e a postura consultiva do SDR."
                }}
                """
                chat2 = executar_chat_com_retentativa(
                    model=MODELO_RAPIDO,
                    messages=[{"role": "system", "content": prompt_agente2}, {"role": "user", "content": texto}],
                    response_format={"type": "json_object"}
                )
                res2 = json.loads(clean_json(chat2.choices[0].message.content))
                time.sleep(4)

                # --------------------------------------------------
                # AGENTE 3: O Diretor de Enablement (Consolidador Técnico Llama 3.3 70B)
                # --------------------------------------------------
                print(" -> Executando Agente 3: Diagnóstico de Impacto e Feedbacks Imediatos...")
                contexto_sintese = f"""
                Resultados do Agente 1 (Processos): {json.dumps(res1)}
                Resultados do Agente 2 (SPIN): {json.dumps(res2)}
                """
                prompt_agente3 = """
                Você é o Agente 3: Diretor de Enablement e Performance Comercial. Sua missão é consolidar os relatórios dos agentes anteriores e gerar um feedback de alto impacto, cirúrgico e direto para o SDR.
                Seja extremamente franco e firme. Não amacie o feedback. Mostre exatamente a ferida operacional do vendedor.

                Você DEVE estruturar sua resposta OBRIGATORIAMENTE usando exatamente estes tópicos formatados em Markdown:

                ### 1. PARECER E POSTURA CONSULTIVA
                [Descreva o diagnóstico macro da postura, se demonstrou autoridade ou se agiu como um mero atendente reativo]

                ### 2. O QUE ERROU
                - [Aponte de forma direta e sem rodeios os desvios cometidos, os dados ignorados ou as dores que aceitou sem aprofundar]

                ### 3. COMO DEVERIA TER FEITO
                - [Apresente exemplos práticos de roteiro e perguntas de follow-up assertivas que o SDR deveria ter aplicado neste caso real]

                ### 4. CAUSA E EFEITO NO FUNIL DE VENDAS
                - [Explique matematicamente o prejuízo operacional gerado por essa falha (ex: pipeline inflado, closer perdendo tempo, queda na conversão de SQL para Fechamento, reuniões superficiais)]

                Responda estritamente neste formato JSON:
                {{
                  "parecer_executivo": "Texto completo contendo os 4 tópicos estruturados exatamente com seus títulos em Markdown como exigido no prompt acima.",
                  "plano_de_acao_curto": "Direcionamento prático e direto para o próximo contato do SDR."
                }}
                """
                chat3 = executar_chat_com_retentativa(
                    model=MODELO_PARERES,
                    messages=[
                        {"role": "system", "content": prompt_agente3},
                        {"role": "user", "content": f"Contexto dos Agentes:\n{contexto_sintese}\n\nTranscrição:\n{texto}"}
                    ],
                    response_format={"type": "json_object"}
                )
                res3 = json.loads(clean_json(chat3.choices[0].message.content))

                # --------------------------------------------------
                # 4. CONSOLIDAÇÃO E SALVAMENTO DOS DADOS NO DATABASE
                # --------------------------------------------------
                s_spin = res2.get("spin_scores", {})
                nota_spin = sum([safe_float(s_spin.get(k)) for k in ['s','p','i','n']]) / 4.0
                nota_op = calcular_nota_operacional(res1.get("operacional", {}), res1.get("erro_fatal", False))
                
                urgencia = "SIM" if (nota_op <= 5.0 or nota_spin <= 5.0 or res1.get("erro_fatal")) else "NÃO"

                db[call_id] = {
                    "id": call_id, "sdr": sdr_name, "produto": produto_detectado, "data": date_str, "duracao": duration,
                    "wps": wps, "nota_spin": round(nota_spin, 1), "nota_op": round(nota_op, 1),
                    "urgencia": urgencia, "deal_url": deal_url, "audio_url": audio_url,
                    "notas_s_p_i_n": s_spin, "formulario": res1.get("operacional", {}),
                    "parecer": res3.get("parecer_executivo", ""), "sugestoes": res3.get("plano_de_acao_curto", ""),
                    "transcricao": texto
                }
                
                with open(CONSOLIDATED_FILE, 'w', encoding='utf-8', errors='ignore') as sf:
                    json.dump(db, sf, ensure_ascii=False, indent=4)
                
                print(f"✅ Auditoria Finalizada! SPIN: {nota_spin:.1f} | Conformidade: {nota_op:.1f} | Alerta: {urgencia}")
                time.sleep(5) 

            except Exception as e:
                print(f"❌ Erro Crítico isolado no ID {call_id}: {e}")
                traceback.print_exc()
                time.sleep(4)

if __name__ == "__main__":
    process_all_calls()
