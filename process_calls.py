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
            return int(partes[0]) * 3600 + int(partes[1]) * 60 + int(partes[2])
        if len(partes) == 2: 
            return int(partes[0]) * 60 + int(partes[1])
    except: 
        pass
    return 1

def calcular_nota_operacional(op_data, erro_fatal):
    """
    MATEMÁTICA FLEXÍVEL E JUSTA DE AUDITORIA:
    - Fim do Zero Absoluto: Erros fatais tiram muitos pontos (-4.0), mas não zeram toda a ligação caso o SDR tenha acertado outras coisas.
    - Foco na Escuta Ativa (N/A): O 'N/A' não penaliza mais o SDR em habilidades estratégicas ou formais, pois significa que o lead entregou a resposta proativamente e o SDR não precisou perguntar.
    """
    nota_final = 10.0

    if erro_fatal: 
        nota_final -= 4.0  # Penalidade severa, mas permite que o SDR pontue nos acertos.

    chaves_criticas = ['sla', 'passos_ro', 'gestao']
    chaves_estrategicas = ['spin', 'dor', 'validacao', 'objecoes', 'produto', 'escuta', 'compreensao']
    chaves_formais = ['linguagem', 'receptividade', 'rapport', 'discurso', 'compreensao_cliente', 'clareza', 'gatilhos']

    # CRÍTICOS: Quebra de processo (SLA, Agendamento) -> Punição Média-Alta
    for k in chaves_criticas:
        r = op_data.get(k, {}).get('r')
        if r == 'Não': 
            nota_final -= 1.5
        elif r == 'N/A': 
            nota_final -= 0.5  # Omissão de dados vitais ainda perde 0.5

    # ESTRATÉGICOS: Construção de valor -> Punição Média (N/A isento)
    for k in chaves_estrategicas:
        r = op_data.get(k, {}).get('r')
        if r == 'Não': 
            nota_final -= 0.8
        # N/A não tira pontos aqui (ex: Lead não teve objeções ou já validou o problema antes de forma clara)

    # FORMAIS: Soft skills e Etiqueta -> Punição Leve (N/A isento)
    for k in chaves_formais:
        r = op_data.get(k, {}).get('r')
        if r == 'Não': 
            nota_final -= 0.4
        # N/A não tira pontos aqui

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
            print(f"🔥 INICIANDO AUDITORIA | ID: {call_id} | SDR: {sdr_name}")
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
                # AGENTE 1: CONFORMIDADE COM N/A PROATIVO
                # --------------------------------------------------
                print(" -> Agente 1: Analisando Conformidade e Adaptação...")
                prompt_agente1 = f"""
                Você é o Agente 1: Auditor Comercial Inteligente. Avalie o SDR no produto {produto_detectado}.
                
                MUITO IMPORTANTE - USO ESTRATÉGICO DO N/A (FLEXIBILIDADE E ESCUTA ATIVA):
                Se uma pergunta do processo de venda não precisou ser feita porque o Lead JÁ FORNECEU a informação de forma proativa na conversa (ex: o lead já disse espontaneamente qual era o problema ou quantos contratos ele tem), marque OBRIGATORIAMENTE "N/A" na validação ou compreensão. NUNCA marque "Não" se o SDR teve a inteligência de ouvir o lead atentamente e não repetir perguntas desnecessárias.

                DIRETRIZES DE AUDITORIA:
                [1. ESCUTA E ADAPTAÇÃO]
                - escuta: O SDR adaptou a conversa? Se interrompeu o lead ou ignorou uma dor para ler o script passivamente, marque "Não".
                - validacao: Marque "N/A" se o lead foi tão claro que a validação não foi necessária. Marque "Não" se o SDR mudou de assunto secamente após o lead confessar um problema grave.
                - compreensao: Inteligência de fluxo. Marque "Não" APENAS se o SDR perguntou de novo algo que o lead já tinha respondido antes, demonstrando desatenção.
                - objecoes: Contornou barreiras? Se o lead não apresentou nenhuma objeção durante a call, marque OBRIGATORIAMENTE "N/A".

                [2. COMUNICAÇÃO E POSTURA B2B]
                - linguagem: Norma culta. ATENÇÃO: Se o SDR usou diminutivos infantis e antiprofissionais (sisteminha, minutinho, propostinha, tempinho), marque "Não".
                - receptividade: Executou a saudação completa de forma acolhedora?
                - rapport: Aproveitou o contexto trazido pelo cliente para quebrar o gelo ou foi um robô?
                - discurso: Usou vocabulário técnico correto do mercado imobiliário e soou como um especialista?
                - compreensao_cliente: Validou com perguntas se o lead estava acompanhando a explicação técnica?
                - clareza: Fez perguntas curtas e diretas ou confundiu o cliente?

                [3. PROCESSO E QUALIFICAÇÃO]
                - sla: 
                  * Para {produto_detectado} CRM: Coletou Número de Corretores E Situação do CRECI?
                  * Para {produto_detectado} ERP: Coletou Quantidade de Contratos E Bancos operados?
                  (Se o lead já entregou a informação sozinho na fala dele sem precisar ser perguntado, marque "Sim" ou "N/A").
                - spin: Seguiu a sequência exploratória de investigação ou só apresentou funcionalidades como um panfleto?
                - dor: Encontrou um gargalo real? Se o lead deu respostas vazias e o SDR não insistiu para descobrir a verdade, marque "Não".
                - gestao: Mapeou e descobriu quem toma a decisão final?
                - passos_ro: Conseguiu a confirmação VERBAL CLARA de que o lead estará num COMPUTADOR na próxima reunião? Se o SDR aceitou um "vou ver pelo celular" ou "estarei no carro", marque "Não".
                - produto: Conectou a solução tecnológica à dor do cliente de forma inteligente?
                - gatilhos: Gerou valor e urgência de agenda para o próximo agendamento?

                REGRAS DE ERRO FATAL E JSON: 
                - Marque "erro_fatal": true APENAS se o SDR quebrar o sigilo e passar preço ou agendar reunião com lead fora de perfil.
                - 🚨 NUNCA use aspas duplas (") dentro das suas frases de "Evidência". Use sempre aspas simples (').

                Retorne OBRIGATORIAMENTE o JSON preenchendo "r" com "Sim", "Não" ou "N/A":
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
                time.sleep(3)

                # --------------------------------------------------
                # AGENTE 2: SPIN COM AVALIAÇÃO JUSTA E FLEXÍVEL
                # --------------------------------------------------
                print(" -> Agente 2: Avaliando Notas de Metodologia SPIN...")
                prompt_agente2 = """
                Você é o Agente 2: Especialista em Metodologia de Vendas e Psicologia Comercial.
                
                INSTRUÇÕES DE NOTAS (SEJA JUSTO E FLEXÍVEL NAS NOTAS MÉDIAS):
                - Notas 9.0 a 10.0: Seja extremamente rigoroso. Só dê nota máxima se o SDR foi cirúrgico, tocou na ferida do cliente e gerou uma urgência inquestionável usando perguntas de Implicação e Necessidade maravilhosas.
                - Notas 5.0 a 8.5: SEJA FLEXÍVEL. Se o SDR tentou investigar, fez perguntas para identificar o problema e manteve a conversa fluindo de forma minimamente investigativa (mesmo que não tenha sido o SPIN perfeito dos livros), dê notas intermediárias boas para recompensar e validar o esforço técnico.
                - Notas 0.0 a 4.5: Use apenas se o SDR foi totalmente reativo, raso ou apenas leu perguntas engessadas de "Situação" como um robô, sem criar nenhum tipo de valor para a dor do cliente.

                🚨 REGRA DE FORMATAÇÃO: NUNCA use aspas duplas (") na sua justificativa, pois quebra o JSON. Use apenas aspas simples (').

                Responda estritamente neste formato JSON:
                {{
                  "spin_scores": {{"s": 5.0, "p": 6.5, "i": 4.0, "n": 3.0}},
                  "analise_autoridade": "Breve justificativa técnica avaliando a postura do vendedor usando aspas simples se precisar."
                }}
                """
                chat2 = executar_chat_com_retentativa(
                    model=MODELO_RAPIDO, 
                    messages=[{"role": "system", "content": prompt_agente2}, {"role": "user", "content": texto}], 
                    response_format={"type": "json_object"}
                )
                res2 = json.loads(clean_json(chat2.choices[0].message.content))
                time.sleep(3)

                # --------------------------------------------------
                # AGENTE 3: FEEDBACK TÁTICO, DIRETO E SEM DESCULPAS GENÉRICAS
                # --------------------------------------------------
                print(" -> Agente 3: Construindo Feedback Técnico Estruturado...")
                contexto_sintese = f"Resultados Agente 1: {json.dumps(res1)}\nResultados Agente 2: {json.dumps(res2)}"
                prompt_agente3 = """
                Você é o Diretor de Enablement. Sua missão é dar feedback técnico para o vendedor de forma absurdamente prática, útil e aplicável.

                🚨 REGRA DE OURO INQUEBRÁVEL (TOLERÂNCIA ZERO PARA FEEDBACK GENÉRICO E PALESTRAS DE IA):
                É EXPRESSAMENTE PROIBIDO usar palavras vazias e burocráticas como "você não seguiu o playbook", "você ignorou o roteiro", "faltou sequência lógica" ou "não seguiu as diretrizes". 
                Se você apontar um erro, VOCÊ DEVE OBRIGATORIAMENTE FORNECER A FALA EXATA que o vendedor deveria ter usado no lugar, como um treinador entregando uma receita prática de vendas.

                🚨 REGRA CRÍTICA DE FORMATAÇÃO JSON (ANTI-ERRO):
                É EXPRESSAMENTE PROIBIDO usar aspas duplas (") dentro dos seus textos gerados. O uso de aspas duplas quebra a estrutura do banco de dados JSON.
                Se precisar citar uma fala do cliente ou do vendedor, use APENAS ASPAS SIMPLES ('). Exemplo correto: o cliente disse 'foi uma porcaria'.

                Estruture SUA resposta OBRIGATORIAMENTE com estes tópicos em Markdown:

                ### 1. PARECER E POSTURA CONSULTIVA
                [Um resumo direto de 2 linhas sobre o controle de conversa e inteligência comercial demonstrada na ligação]

                ### 2. O QUE ERROU
                - [Aponte falhas REAIS encontradas na transcrição. Ex: "No minuto 03:10, o cliente disse que perde horas fechando o caixa, mas você não aprofundou e mudou de assunto perguntando de sistema atual."]

                ### 3. COMO DEVERIA TER FEITO (SCRIPT PRÁTICO)
                - [Obrigatoriamente forneça o texto exato em formato de fala. Ex: "Em vez de mudar de assunto, você deveria ter ancorado a dor e perguntado: 'Cliente, se você perde todo esse tempo fechando o caixa, como fica o seu repasse para os proprietários no final do mês?'"]
                *Aviso: Consulte a aba 'Playbooks SPIN' no menu lateral do sistema operacional para revisar a estrutura de perguntas abertas e fechadas.*

                ### 4. CAUSA E EFEITO NO FUNIL DE VENDAS
                - [Explique de forma direta e rápida como esse erro específico na call acaba esfriando o lead e prejudicando a taxa de conversão na Demonstração com o Closer.]

                Responda estritamente neste formato JSON:
                {{
                  "parecer_executivo": "O texto completo respeitando fielmente os títulos em Markdown solicitados acima e sem usar aspas duplas.",
                  "plano_de_acao_curto": "A pergunta exata ou a postura única que ele deve treinar para a próxima ligação, sem usar aspas duplas."
                }}
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
                # 4. CONSOLIDAÇÃO DOS DADOS NO ARQUIVO
                # --------------------------------------------------
                s_spin = res2.get("spin_scores", {})
                nota_spin = sum([safe_float(s_spin.get(k)) for k in ['s','p','i','n']]) / 4.0
                nota_op = calcular_nota_operacional(res1.get("operacional", {}), res1.get("erro_fatal", False))
                
                urgencia = "SIM" if (nota_op <= 5.0 or nota_spin <= 5.0) else "NÃO"

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
                
                print(f"✅ Auditoria Finalizada com Sucesso! SPIN: {nota_spin:.1f} | Conformidade: {nota_op:.1f}")
                time.sleep(4)

            except Exception as e:
                print(f"❌ Erro na auditoria do ID {call_id}: {e}")
                traceback.print_exc()
                time.sleep(3)

if __name__ == "__main__":
    process_all_calls()
