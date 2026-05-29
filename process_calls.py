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
PORTAL_ID = "20131994"

# ==========================================
# 2. FUNÇÕES AUXILIARES E DE SEGURANÇA
# ==========================================
def clean_json(text):
    """Extrai apenas o JSON da resposta da IA, removendo formatações markdown."""
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
    if erro_fatal: 
        return 0.0

    b1_chaves = ['escuta', 'validacao', 'compreensao', 'objecoes']
    b1_sim = sum(1 for k in b1_chaves if op_data.get(k, {}).get('r') == 'Sim')
    nota_b1 = (b1_sim / 4.0) * 100.0

    b2_chaves = ['linguagem', 'receptividade', 'rapport', 'discurso', 'compreensao_cliente', 'clareza']
    b2_sim = sum(1 for k in b2_chaves if op_data.get(k, {}).get('r') == 'Sim')
    nota_b2 = (b2_sim / 6.0) * 100.0

    b3_chaves = ['sla', 'spin', 'dor', 'gestao', 'passos_ro', 'produto', 'gatilhos']
    b3_sim = sum(1 for k in b3_chaves if op_data.get(k, {}).get('r') == 'Sim')
    b3_na = sum(1 for k in b3_chaves if op_data.get(k, {}).get('r') == 'N/A')
    nota_b3 = (b3_sim * 14.3) + (b3_na * 2.0)

    nota_final = (nota_b1 + nota_b2 + nota_b3) / 30.0
    return min(max(nota_final, 0.0), 10.0)

# ==========================================
# 3. NÚCLEO DE PROCESSAMENTO
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
        except Exception as e: 
            print(f"Base de dados limpa ou vazia. A criar novo registo.")
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
            
            # Construir o Deal URL usando a coluna Associated Deal IDs
            deal_id = row.get("Associated Deal IDs", "").strip()
            deal_url = ""
            if deal_id:
                primeiro_id = deal_id.split(',')[0].strip()
                deal_url = f"[https://app.hubspot.com/contacts/](https://app.hubspot.com/contacts/){PORTAL_ID}/deal/{primeiro_id}/"

            if not call_id or not audio_url.startswith("http") or result.lower() not in ["ligação atendida", "connected", "atendida"]:
                continue
                
            if call_id in db:
                continue

            print(f"\n-> [AUDITANDO] ID {call_id} | {sdr_name}...")
            
            txt_verif = (title + " " + json.dumps(row)).lower()
            produto_detectado = "CRM" if any(p in txt_verif for p in ["crm", "creci", "corretor"]) else "ERP"

            try:
                # 1. Obter o áudio
                req = urllib.request.Request(audio_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=45) as response:
                    audio_bytes = response.read()

                # 2. Whisper (Transcrição)
                transcription = client.audio.transcriptions.create(
                    file=("audio.mp3", io.BytesIO(audio_bytes)),
                    model="whisper-large-v3",
                    response_format="json"
                )
                texto = transcription.text
                if len(texto) < 10: 
                    print(f"Ignorado: Áudio demasiado curto.")
                    continue

                # 3. Calcular WPS
                segundos = calcular_segundos(duration)
                wps = round(len(texto.split()) / segundos, 2) if segundos > 0 else 0.0

                # 4. Auditoria com LLM
                prompt = f"""
                És o Auditor Sênior de Vendas da Superlógica. Avalia esta transcrição ({produto_detectado}).
                
                REGRAS FATAIS:
                - ERP: Se falar de Preço ou o lead tiver 0 contratos = ERRO FATAL (marque erro_fatal como true).
                - CRM: Se falar de Preço ou o lead não tiver CRECI = ERRO FATAL (marque erro_fatal como true).
                
                Para cada critério "operacional", usa "Sim", "Não" ou "N/A" no campo "r" e justifique curto em "e".
                
                FORMATAÇÃO OBRIGATÓRIA JSON:
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
                  "parecer_tecnico": "Escreve um feedback profundo sobre as falhas e acertos da chamada.",
                  "plano_acao": "Define os pontos de melhoria e as ações práticas que o gestor deve aplicar ao SDR."
                }}
                """

                chat = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"Transcrição:\n{texto}"}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                
                raw_response = chat.choices[0].message.content
                cleaned_response = clean_json(raw_response)
                dados_ia = json.loads(cleaned_response)
                
                # 5. Calcular Notas Finais
                s_spin = dados_ia.get("spin", {})
                nota_spin = sum([safe_float(s_spin.get(k)) for k in ['s','p','i','n']]) / 4.0
                nota_op = calcular_nota_operacional(dados_ia.get("operacional", {}), dados_ia.get("erro_fatal", False))
                
                urgencia = "SIM" if (nota_op <= 5.0 or nota_spin <= 5.0 or dados_ia.get("erro_fatal")) else "NÃO"

                # 6. Guardar no Banco de Dados JSON
                db[call_id] = {
                    "id": call_id, "sdr": sdr_name, "produto": produto_detectado, "data": date_str, "duracao": duration,
                    "wps": wps, "nota_spin": round(nota_spin, 1), "nota_op": round(nota_op, 1),
                    "urgencia": urgencia, "deal_url": deal_url, "audio_url": audio_url,
                    "notas_s_p_i_n": s_spin, "formulario": dados_ia.get("operacional", {}),
                    "parecer": dados_ia.get("parecer_tecnico", ""), "sugestoes": dados_ia.get("plano_acao", ""),
                    "transcricao": texto
                }
                
                with open(CONSOLIDATED_FILE, 'w', encoding='utf-8') as sf: 
                    json.dump(db, sf, ensure_ascii=False, indent=4)
                
                print(f"✅ Concluído! SPIN: {nota_spin:.1f} | OP: {nota_op:.1f} | WPS: {wps}")
                time.sleep(2)

            except Exception as e:
                print(f"❌ Erro no Processamento do ID {call_id}: {e}")
                time.sleep(3)

if __name__ == "__main__":
    process_all_calls()
