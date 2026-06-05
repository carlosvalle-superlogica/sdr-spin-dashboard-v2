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
    MATEMÁTICA DE AUDITORIA DE TOLERÂNCIA ZERO COM PESOS DE GRAVIDADE:
    - O SDR inicia com a nota máxima de 10.0 (Exigência de perfeição).
    - O 'Sim' mantém a nota estável.
    - As 17 chaves originais são divididas por impacto real no funil de vendas.
    - O 'Não' e o 'N/A' penalizam proporcionalmente de acordo com o Tier do critério.
    - A chave 'objecoes' em N/A continua totalmente isenta de punição (venda lisa).
    """
    if erro_fatal: 
        return 0.0

    # Divisão estratégica das 17 chaves originais por Tiers de Impacto
    chaves_criticas = ['sla', 'passos_ro', 'gestao']
    chaves_estrategicas = ['spin', 'dor', 'validacao', 'objecoes', 'produto', 'escuta', 'compreensao']
    chaves_formais = ['linguagem', 'receptividade', 'rapport', 'discurso', 'compreensao_cliente', 'clareza', 'gatilhos']

    nota_final = 10.0

    # 1. Penalidades para Critérios Críticos (Mata a agenda do Closer se falhar)
    for k in chaves_criticas:
        r = op_data.get(k, {}).get('r')
        if r == 'Não':
            nota_final -= 2.0  # Punição severa por quebra ativa de processo
        elif r == 'N/A':
            nota_final -= 1.0  # Punição por omissão ou falta de coleta do dado vital

    # 2. Penalidades para Critérios Estratégicos (Afeta a construção de valor e qualificação)
    for k in chaves_estrategicas:
        r = op_data.get(k, {}).get('r')
        if r == 'Não':
            nota_final -= 1.0
        elif r == 'N/A':
            if k != 'objecoes':  # Preserva a trava de segurança de objeções em N/A
                nota_final -= 0.5

    # 3. Penalidades para Critérios Formais (Soft Skills, Etiqueta e Postura B2B)
    for k in chaves_formais:
        r = op_data.get(k, {}).get('r')
        if r == 'Não':
            nota_final -= 0.5
        elif r == 'N/A':
            nota_final -= 0.25

    # Clampa rigidamente o resultado final entre o piso de 0.0 e o teto de 10.0
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
                Você é o Agente 1: Auditor de Processos Inflexível, Rígido e Analítico. Sua missão é analisar a transcrição da chamada e julgar o cumprimento exato das políticas comerciais, de qualificação e de postura B2B da empresa para o produto {produto_detectado}.
                Seja agressivo, extremamente criterioso e intolerante com falhas. Menções vagas, superficiais ou desculpas aceitas passivamente pelo SDR DEVEM receber a marcação "Não".

                DIRETRIZES DE AUDITORIA (FOCO ESTRITO EM CONFORMIDADE DE ROTEIRO E PLAYBOOK):
                
                [1. ESCUTA E ADAPTAÇÃO]
                - escuta: Avalie se o SDR adaptou a ordem das perguntas do roteiro com base nas falas espontâneas do cliente. Se o SDR interrompeu o lead ou ignorou uma informação útil para apenas seguir lendo o texto, marque "Não".
                - validacao: O SDR aplicou a técnica de ancoragem prevista no roteiro para confirmar o problema? (Ex de roteiro obrigatório: "Deixa eu ver se entendi bem, o seu principal gargalo hoje é X, correto?").
                - compreensao: Inteligência de fluxo. Marque "Não" se o SDR fez alguma pergunta do roteiro cujo dado o cliente já havia entregue espontaneamente antes. Isso demonstra leitura mecânica.
                - objecoes: O SDR utilizou a matriz de contorno de objeções técnica do playbook para responder às travas do lead (tempo, sistema atual, financeiro)? Marque "Não" se ele aceitou o recuo do cliente sem contra-argumentar. Marque "N/A" apenas se a chamada não teve objeções.

                [2. COMUNICAÇÃO E POSTURA B2B]
                - linguagem: Uso da norma culta e ausência de vícios recorrentes. ATENÇÃO: Se o SDR violou o manual de termos proibidos e usou QUALQUER diminutivo para falar do produto ou processo (ex: sisteminha, minutinho, propostinha, conversinha), marque "Não" imediatamente por quebrar a postura corporativa madura.
                - receptividade: Execução do script de abertura. O SDR seguiu a estrutura mandatória de saudação da empresa (Identificação pessoal + Identificação da Superlógica + Gancho do motivo do contato)?
                - rapport: O SDR aproveitou o contexto inicial trazido pelo lead para fazer uma quebra de gelo contextualizada ou agiu como um robô frio de telemarketing?
                - discurso: Domínio dos termos do mercado. O SDR aplicou corretamente o vocabulário técnico do playbook (Repasse, Inadimplência, DIMOB, Leads, CRM, Captação) de forma natural?
                - compreensao_cliente: O SDR executou as perguntas de validação de clareza previstas no roteiro após explicar uma funcionalidade (ex: "Faz sentido essa dinâmica na sua imobiliária?", "Conseguiu visualizar esse processo?")?
                - clareza: Estrutura das perguntas. O SDR fez perguntas curtas e diretas conforme o playbook ou formulou perguntas duplas/confusas que fizeram o cliente pedir para repetir?

                [3. PROCESSO E POLÍTICAS DE QUALIFICAÇÃO]
                - sla: Extração obrigatória de dados de qualificação por linha de produto.
                  * Para {produto_detectado} CRM: O roteiro exige coletar Número de Corretores E Situação do CRECI.
                  * Para {produto_detectado} ERP: O roteiro exige coletar Quantidade de Contratos Ativos E Bancos operados.
                  Se o SDR pulou a coleta de qualquer um dos dois dados exigidos para o produto correspondente, marque "Não".
                - spin: O SDR seguiu a sequência lógica de investigação do playbook (Situação -> Problema -> Implicação) ou limitou-se a fazer um monólogo institucional sobre a Superlógica?
                - dor: Identificação de gargalo técnico. O SDR seguiu o roteiro de investigação até obter a confirmação de um problema real na operação, ou aceitou respostas evasivas ("está tudo ótimo", "só quero conhecer") sem insistir ou aplicar uma redundância para expor a dor real? Se aceitou a enrolação passivamente, marque "Não".
                - gestao: O SDR executou o bloco de mapeamento de organograma para descobrir quem é o decisor final ou se existem outros sócios/diretores envolvidos no processo de escolha?
                - passos_ro: Alinhamento estrito de expectativas para a Demonstração. O SDR exigiu e obteve o compromisso verbal explícito do lead de que ele estará sentado na frente de um COMPUTADOR ou NOTEBOOK? Se o lead sugeriu ver pelo celular ou dirigir durante a call e o SDR aceitou, marque "Não".
                - produto: O SDR conectou a solução do {produto_detectado} como resposta direta à dor específica que o lead confessou, respeitando os ganchos de valor do manual?
                - gatilhos: O SDR aplicou os gatilhos de escassez de agenda ou urgência previstos no script para valorizar o horário com o especialista técnico?

                REGRAS DE VIOLAÇÃO DE POLÍTICA (ERRO FATAL):
                Você DEVE marcar "erro_fatal": true se o SDR cometer qualquer uma das infrações abaixo:
                1. PREÇO: Se o SDR violou a política de sigilo de valores e passou preços, tabelas, estimativas financeiras, taxas ou valores mensais (ex: "custa a partir de R$...", "fica na faixa de X").
                2. AGENDAMENTO DE LEAD INVÁLIDO: Se o SDR aceitou agendar a reunião com um lead que NÃO possui os critérios mínimos do produto (Para CRM: lead com 0 corretores ou sem CRECI definitivo/em andamento. Para ERP: lead com 0 contratos ativos).

                Retorne OBRIGATORIAMENTE este formato JSON preenchendo o campo "r" APENAS com "Sim", "Não" ou "N/A":
                {{
                  "erro_fatal": false,
                  "operacional": {{
                    "escuta": {{"r": "[Sim/Não/N/A]", "e": "Evidência exata extraída da transcrição"}},
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
                # AGENTE 3: O Diretor de Enablement (Consolidador de Roteiro Llama 3.3 70B)
                # --------------------------------------------------
                print(" -> Executando Agente 3: Diagnóstico de Impacto e Feedbacks Imediatos...")
                contexto_sintese = f"""
                Resultados do Agente 1 (Processos): {json.dumps(res1)}
                Resultados do Agente 2 (SPIN): {json.dumps(res2)}
                """
                prompt_agente3 = """
                Você é o Agente 3: Diretor de Enablement e Performance Comercial. Sua missão é consolidar os relatórios dos agentes anteriores e gerar um feedback de alto impacto, cirúrgico e direto focado na CORREÇÃO DO ROTEIRO E DO PLAYBOOK comercial.
                Foque estritamente na execução técnica das etapas do processo de qualificação e no roteiro. Não faça puxadas de orelha comportamentais ou julgamentos subjetivos sobre o tom emocional do vendedor. Mostre exatamente qual linha do manual de vendas foi violada ou ignorada.

                Você DEVE estruturar sua resposta OBRIGATORIAMENTE usando exatamente estes tópicos formatados em Markdown:

                ### 1. PARECER E POSTURA CONSULTIVA
                [Descreva o diagnóstico macro da postura técnica e aderência ao playbook de processos da empresa]

                ### 2. O QUE ERROU
                - [Aponte de forma direta, com marcação de tempo ou citação, os desvios cometidos em relação ao roteiro, os dados do SLA pulados ou as dores ignoradas]

                ### 3. COMO DEVERIA TER FEITO
                - [Apresente exemplos práticos de roteiro adaptado, roteiros de contra-argumentação de objeções e perguntas de follow-up assertivas do manual que o SDR deveria ter aplicado neste caso real]

                ### 4. CAUSA E EFEITO NO FUNIL DE VENDAS
                - [Explique matematicamente o prejuízo operacional gerado por essa quebra de processo técnico no pipeline (ex: pipeline inflado com leads frios, perda de conversão de SQL para Fechamento, Closers perdendo tempo de demonstração)]

                Responda estritamente neste formato JSON:
                {{
                  "parecer_executivo": "Texto completo contendo os 4 tópicos estruturados exatamente com seus títulos em Markdown como exigido no prompt acima.",
                  "plano_de_acao_curto": "Direcionamento prático de roteiro e pergunta exata a ser aplicada no próximo contato do SDR."
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
