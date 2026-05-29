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
# 1. CONFIGURAÇÕES E VARIÁVEIS
# ==========================================
GROQ_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    raise ValueError("ERRO CRÍTICO: GROQ_API_KEY não encontrada nos Secrets!")

client = Groq(api_key=GROQ_KEY)
CSV_FILE = "dados_chamadas.csv"
CONSOLIDATED_FILE = "consolidated_data.json"

# ==========================================
# 2. FUNÇÕES AUXILIARES E DE SEGURANÇA
# ==========================================
def clean_json(text):
    """Força a extração apenas do bloco JSON, ignorando textos que a IA possa gerar antes ou depois."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text
    except:
        return text

def safe_float(val, default=0.0):
    try: 
        return float(val)
    except: 
        return default

def calcular_segundos(duracao_str):
    try:
        partes = duracao_str.split(':')
        if len(partes) == 3: return int(partes[0])*3600 + int(partes[1])*60 + int(partes[2])
        if len(partes) == 2: return int(partes[0])*60 + int(partes[1])
    except: 
        pass
    return 1 # Evita divisão por zero

def calcular_nota_operacional(op_data, erro_fatal):
    if erro_fatal: 
        return 0.0

    # Bloco 1: 4 itens (25% cada)
    b1_chaves = ['escuta', 'validacao', 'compreensao', 'objecoes']
    b1_sim = sum(1 for k in b1_chaves if op_data.get(k, {}).get('r') == 'Sim')
    nota_b1 = (b1_sim / 4.0) * 100.0

    # Bloco 2: 6 itens (16.6% cada)
    b2_chaves = ['linguagem', 'receptividade', 'rapport', 'discurso', 'compreensao_cliente', 'clareza']
    b2_sim = sum(1 for k in b2_chaves if op_data.get(k, {}).get('r') == 'Sim')
    nota_b2 = (b2_sim / 6.0) * 100.0

    # Bloco 3: 7 itens (14.3% por Sim, +2 pontos por N/A)
    b3_chaves = ['sla', 'spin', 'dor', 'gestao', 'passos_ro', 'produto', 'gatilhos']
    b3_sim = sum(1 for k in b3_chaves if op_data.get(k, {}).get('r') == 'Sim')
    b3_na = sum(1 for k in b3_chaves if op_data.get(k, {}).get('r') == 'N/A')
    nota_b3 = (b3_sim * 14.3) + (b3_na * 2.0)

    # Média dos 3 blocos convertida para escala 0 a 10
    nota_final = (nota_b1 + nota_b2 + nota_b3) / 30.0
    return min(max(nota_final, 0.0), 10.0)

# ==========================================
# 3. NÚCLEO DE PROCESSAMENTO
# ==========================================
def process_all_calls():
    if not os.path.exists(CSV_FILE): 
        print(f"Erro: Arquivo {CSV_FILE} não encontrado.")
        return
        
    db = {}
    if os.path.exists(CONSOLIDATED_FILE):
        try:
            with open(CONSOLIDATED_FILE, 'r', encoding='utf-8') as f: 
                db = json.load(f)
        except Exception as e: 
            print(f"Aviso ao ler DB: {e}. Criando novo DB.")
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
            
            # Buscar link do negócio dinamicamente
            deal_url = ""
            for k in row.keys():
                if k and "Chamada" in k and "Negócio" in k:
                    deal_url = row.get(k, "").strip()
                    break

            if not call_id or not audio_url.startswith("http") or result.lower() not in ["ligação atendida", "connected", "atendida"] or call_id in db:
                continue

            print(f"\n-> [AUDITANDO] ID {call_id} | {sdr_name}...")
            
            # Detectar Produto (CRM vs ERP)
            txt_verif = (title + " " + json.dumps(row)).lower()
            produto_detectado = "CRM" if any(p in txt_verif for p in ["crm", "creci", "corretor"]) else "ERP"

            try:
                # 1. Download do Áudio
                req = urllib.request.Request(audio_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=45) as response:
                    audio_bytes = response.read()

                # 2. Transcrição (Whisper)
                transcription = client.audio.transcriptions.create(
                    file=("audio.mp3", io.BytesIO(audio_bytes)),
                    model="whisper-large-v3",
                    response_format="json"
                )
                texto = transcription.text
                if len(texto) < 10: 
                    print(f"Ignorado: Áudio muito curto ou vazio.")
                    continue

                # 3. Cálculo de Palavras por Segundo (WPS)
                segundos = calcular_segundos(duration)
                wps = round(len(texto.split()) / segundos, 2) if segundos > 0 else 0.0

                # 4. Prompt Blindado para Llama 3.1
                prompt = f"""
                Você é o Auditor de Qualidade Sênior da Superlógica. Avalie esta transcrição de venda do produto {produto_detectado}.
                
                REGRAS DE CORTE:
                - ERP: Preço passado ou Lead com 0 contratos = ERRO FATAL (marque erro_fatal como true).
                - CRM: Preço passado ou Lead sem CRECI = ERRO FATAL (marque erro_fatal como true).
                
                Para cada critério em 'operacional', responda EXATAMENTE com "Sim", "Não" ou "N/A" na chave "r", e uma frase curta de evidência na chave "e".
                
                RESPONDA ESTRITAMENTE NESTE FORMATO JSON:
                {{
                  "erro_fatal": false,
                  "spin": {{"s": 8.0, "p": 7.0, "i": 6.0, "n": 9.0}},
                  "operacional": {{
                    "escuta": {{"r": "Sim", "e": "..."}},
                    "validacao": {{"r": "Não", "e": "..."}},
                    "compreensao": {{"r": "Sim", "e": "..."}},
                    "objecoes": {{"r": "Sim", "e": "..."}},
                    "linguagem": {{"r": "Sim", "e": "..."}},
                    "receptividade": {{"r": "Sim", "e": "..."}},
                    "rapport": {{"r": "Sim", "e": "..."}},
                    "discurso": {{"r": "Sim", "e": "..."}},
                    "compreensao_cliente": {{"r": "Sim", "e": "..."}},
                    "clareza": {{"r": "Sim", "e": "..."}},
                    "sla": {{"r": "Não", "e": "..."}},
                    "spin": {{"r": "Sim", "e": "..."}},
                    "dor": {{"r": "Sim", "e": "..."}},
                    "gestao": {{"r": "Sim", "e": "..."}},
                    "passos_ro": {{"r": "N/A", "e": "..."}},
                    "produto": {{"r": "Sim", "e": "..."}},
                    "gatilhos": {{"r": "Sim", "e": "..."}}
                  }},
                  "parecer_tecnico": "Resumo macro da avaliação...",
                  "plano_acao": "Sugestão prática..."
                }}
                """

                # 5. Chamada de Análise (Groq Llama 3.1)
                chat = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"Transcrição:\n{texto}"}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                
                # 6. Processamento e Limpeza do JSON
                raw_response = chat.choices[0].message.content
                cleaned_response = clean_json(raw_response)
                
                try:
                    dados_ia = json.loads(cleaned_response)
                except json.JSONDecodeError as je:
                    print(f"❌ Erro ao ler JSON da IA: {je}")
                    print(f"Raw Output: {raw_response}")
                    continue # Salta esta chamada se o JSON for inválido
                
                # 7. Matemáticas das Notas
                s_spin = dados_ia.get("spin", {})
                nota_spin = sum([safe_float(s_spin.get(k)) for k in ['s','p','i','n']]) / 4.0
                nota_op = calcular_nota_operacional(dados_ia.get("operacional", {}), dados_ia.get("erro_fatal", False))
                
                # 8. Flag de Urgência
                urgencia = "SIM" if (nota_op <= 5.0 or nota_spin <= 5.0 or dados_ia.get("erro_fatal")) else "NÃO"

                # 9. Guardar na Base de Dados
                db[call_id] = {
                    "id": call_id, "sdr": sdr_name, "produto": produto_detectado, "data": date_str, "duracao": duration,
                    "wps": wps, "nota_spin": round(nota_spin, 1), "nota_op": round(nota_op, 1),
                    "urgencia": urgencia, "deal_url": deal_url, "audio_url": audio_url,
                    "notas_s_p_i_n": s_spin, "formulario": dados_ia.get("operacional", {}),
                    "parecer": dados_ia.get("parecer_tecnico", ""), "sugestoes": dados_ia.get("plano_acao", ""),
                    "transcricao": texto
                }
                
                # Guarda ficheiro incrementalmente para não perder dados se crashar
                with open(CONSOLIDATED_FILE, 'w', encoding='utf-8') as sf: 
                    json.dump(db, sf, ensure_ascii=False, indent=4)
                
                print(f"✅ Concluído! SPIN: {nota_spin:.1f} | OP: {nota_op:.1f} | WPS: {wps}")
                time.sleep(2.5) # Pausa ligeiramente maior para Rate Limit

            except Exception as e:
                print(f"❌ Erro no Processamento do ID {call_id}: {e}")
                traceback.print_exc()
                time.sleep(3)

if __name__ == "__main__":
    process_all_calls()
