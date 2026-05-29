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
# 2. SISTEMAS DE SEGURANÇA E MATEMÁTICA
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
    """Converte strings de duração formatadas (HH:mm:ss ou mm:ss) in segundos totais."""
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
    """Executa a matemática rigorosa de pontuação com o peso do 'Não se Aplica' (+2 pontos)."""
    if erro_fatal: 
        return 0.0

    # Bloco 1: Escuta e Adaptação (4 itens - 25% cada)
    b1_chaves = ['escuta', 'validacao', 'compreensao', 'objecoes']
    b1_sim = sum(1 for k in b1_chaves if op_data.get(k, {}).get('r') == 'Sim')
    nota_b1 = (b1_sim / 4.0) * 100.0

    # Bloco 2: Comunicação Clara (6 itens - 16.6% cada)
    b2_chaves = ['linguagem', 'receptividade', 'rapport', 'discurso', 'compreensao_cliente', 'clareza']
    b2_sim = sum(1 for k in b2_chaves if op_data.get(k, {}).get('r') == 'Sim')
    nota_b2 = (b2_sim / 6.0) * 100.0

    # Bloco 3: Processo de Qualificação (7 itens - 14.3% por Sim, +2.0% por N/A)
    b3_chaves = ['sla', 'spin', 'dor', 'gestao', 'passos_ro', 'produto', 'gatilhos']
    b3_sim = sum(1 for k in b3_chaves if op_data.get(k, {}).get('r') == 'Sim')
    b3_na = sum(1 for k in b3_chaves if op_data.get(k, {}).get('r') == 'N/A')
    nota_b3 = (b3_sim * 14.3) + (b3_na * 2.0)

    # Nota operacional final ponderada escalada de 0 a 10
    nota_final = (nota_b1 + nota_b2 + nota_b3) / 30.0
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

            try:
                # 1. Download do Áudio da Gravação
                req = urllib.request.Request(audio_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=45) as response: 
                    audio_bytes = response.read()

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
                Você é o Agente 1: Auditor de Processos Rígido. Avalie se o SDR seguiu estritamente as regras de negócio de {produto_detectado}.
                Seja agressivo. Só marque "Sim" se houver prova clara e incontestável na transcrição. Menções superficiais devem ser marcadas como "Não".

                REGRAS DE ERRO FATAL (Se violadas, marque "erro_fatal": true):
                - {produto_detectado} ERP: Passar preço ou aceitar lead com 0 contratos. SLA exige quantidade de contratos + bancos.
                - {produto_detectado} CRM: Passar preço ou aceitar lead sem CRECI. SLA exige número de corretores + CRECI.
                - Ambos: Passos RO exige que o lead se comprometa a estar na frente do COMPUTADOR para a reunião com o especialista.

                Responda estritamente neste formato JSON:
                {{
                  "erro_fatal": false,
                  "operacional": {{
                    "escuta": {{"r": "Sim", "e": "Evidência"}},
                    "validacao": {{"r": "Não", "e": "Evidência"}},
                    "compreensao": {{"r": "Sim", "e": "Evidência"}},
                    "objecoes": {{"r": "Sim", "e": "Evidência"}},
                    "linguagem": {{"r": "Sim", "e": "Evidência"}},
                    "receptividade": {{"r": "Sim", "e": "Evidência"}},
                    "rapport": {{"r": "Sim", "e": "Evidência"}},
                    "discurso": {{"r": "Sim", "e": "Evidência"}},
                    "compreensao_cliente": {{"r": "Sim", "e": "Evidência"}},
                    "clareza": {{"r": "Sim", "e": "Evidência"}},
                    "sla": {{"r": "Não", "e": "Evidência"}},
                    "spin": {{"r": "Sim", "e": "Evidência"}},
                    "dor": {{"r": "Sim", "e": "Evidência"}},
                    "gestao": {{"r": "Sim", "e": "Evidência"}},
                    "passos_ro": {{"r": "Não", "e": "Evidência"}},
                    "produto": {{"r": "Sim", "e": "Evidência"}},
                    "gatilhos": {{"r": "Sim", "e": "Evidência"}}
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
                
                with open(CONSOLIDATED_FILE, 'w', encoding='utf-8') as sf:
                    json.dump(db, sf, ensure_ascii=False, indent=4)
                
                print(f"✅ Auditoria Finalizada! SPIN: {nota_spin:.1f} | Conformidade: {nota_op:.1f} | Alerta: {urgencia}")
                time.sleep(5) 

            except Exception as e:
                print(f"❌ Erro Crítico isolado no ID {call_id}: {e}")
                traceback.print_exc()
                time.sleep(4)

if __name__ == "__main__":
    process_all_calls()
