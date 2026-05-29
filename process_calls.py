import os
import csv
import json
import urllib.request
from urllib.error import URLError, HTTPError
from datetime import datetime
from groq import Groq

# ==========================================
# 1. CONFIGURAÇÕES E VARIÁVEIS DE AMBIENTE
# ==========================================
GROQ_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    raise ValueError("ERRO CRÍTICO: GROQ_API_KEY não encontrada nos Secrets do GitHub!")

# Inicializa o cliente do Groq
client = Groq(api_key=GROQ_KEY)

# Lista de possíveis nomes para o arquivo para evitar erro de extensão dupla
CSV_CANDIDATES = ["dados_chamadas.csv", "dados_chamadas.csv.csv"]
PROMPT_FILE = "evaluation_prompt.txt"
ANALYSES_DIR = "analises_salvas"
CONSOLIDATED_FILE = "consolidated_data.json"

# ==========================================
# 2. FUNÇÕES AUXILIARES
# ==========================================
def load_prompt():
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()

def ensure_directories():
    os.makedirs(ANALYSES_DIR, exist_ok=True)

def parse_date_to_year_month(date_str):
    if not date_str:
        return None, None
    try:
        # Suporta formatos "2026-05-28 09:13" ou com "/ "
        clean_date = date_str.split()[0].replace('/', '-')
        if '-' in clean_date:
            parts = clean_date.split('-')
            if len(parts[0]) == 4: # YYYY-MM-DD
                dt = datetime.strptime(clean_date, "%Y-%m-%d")
            else: # DD-MM-YYYY
                dt = datetime.strptime(clean_date, "%d-%m-%Y")
        
        year = str(dt.year)
        month_names = [
            "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", 
            "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
        ]
        month = month_names[dt.month - 1]
        return year, month
    except Exception as e:
        print(f"Erro ao converter data '{date_str}': {e}")
        return None, None

def clean_json_response(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# ==========================================
# 3. NÚCLEO DE PROCESSAMENTO
# ==========================================
def process_all_calls():
    ensure_directories()

    # Encontra qual arquivo de dados está presente
    target_csv = None
    for candidate in CSV_CANDIDATES:
        if os.path.exists(candidate):
            target_csv = candidate
            break

    if not target_csv:
        print(f"Aviso: Nenhum arquivo CSV válido encontrado ({CSV_CANDIDATES}). Encerrando.")
        return

    print(f"✅ Arquivo de dados encontrado: {target_csv}")
    prompt_content = load_prompt()
    
    if os.path.exists(CONSOLIDATED_FILE):
        with open(CONSOLIDATED_FILE, 'r', encoding='utf-8') as f:
            db = json.load(f)
    else:
        db = {}

    with open(target_csv, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Mapeamento Inteligente (Aceita colunas em Português ou Inglês do HubSpot)
            call_id = row.get("Object ID") or row.get("ID do objeto")
            sdr_name = row.get("Activity assigned to") or row.get("Atividade atribuída a") or "SDR Não Identificado"
            date_str = row.get("Activity date") or row.get("Data da atividade")
            audio_url = row.get("Call recording URL") or row.get("URL de gravação")
            result = row.get("Call outcome") or row.get("Resultado da chamada")

            # Filtro: Só analisa chamadas com áudio e válidas
            if not call_id or not audio_url:
                continue
            
            if result not in ["Connected", "Ligação atendida", "Atendida"]:
                continue

            year, month = parse_date_to_year_month(date_str)
            if not year or not month:
                continue

            saved_analysis_path = os.path.join(ANALYSES_DIR, f"{call_id}.json")
            analysis_data = None

            if os.path.exists(saved_analysis_path):
                print(f"-> [CACHE] Ligação {call_id} já analisada.")
                try:
                    with open(saved_analysis_path, 'r', encoding='utf-8') as sf:
                        analysis_data = json.load(sf)
                except Exception:
                    continue
            else:
                print(f"-> [NOVA] Analisando ID {call_id} | SDR: {sdr_name}...")
                
                try:
                    req = urllib.request.Request(
                        audio_url, 
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    )
                    with urllib.request.urlopen(req, timeout=45) as response:
                        audio_bytes = response.read()

                    # ETAPA 1: Transcrição via Whisper
                    transcription = client.audio.transcriptions.create(
                        file=("audio.wav", audio_bytes),
                        model="whisper-large-v3",
                        response_format="json"
                    )
                    texto_ligacao = transcription.text

                    # ETAPA 2: Análise Comercial via Llama 3
                    chat_completion = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {
                                "role": "system", 
                                "content": f"{prompt_content}\n\nIMPORTANTE: Retorne APENAS um objeto JSON válido. Não inclua textos antes ou depois."
                            },
                            {
                                "role": "user", 
                                "content": f"Analise a seguinte transcrição da ligação:\n\n{texto_ligacao}"
                            }
                        ],
                        temperature=0.3
                    )

                    clean_text = clean_json_response(chat_completion.choices[0].message.content)
                    analysis_data = json.loads(clean_text)
                    
                    with open(saved_analysis_path, 'w', encoding='utf-8') as sf:
                        json.dump(analysis_data, sf, ensure_ascii=False, indent=2)

                except Exception as e:
                    print(f"Erro na chamada {call_id}: {e}")
                    continue

            if analysis_data:
                if year not in db: db[year] = {}
                if month not in db[year]: db[year][month] = {}
                if sdr_name not in db[year][month]: db[year][month][sdr_name] = []

                call_entry = {
                    "id_registro": call_id,
                    "data_atividade": date_str,
                    "titulo": row.get("Call title") or row.get("Título da chamada", "Chamada de Vendas"),
                    "duracao": row.get("Call duration") or row.get("Duração da chamada (HH:mm:ss)", "00:00"),
                    "analise": analysis_data
                }
                
                if not any(item["id_registro"] == call_id for item in db[year][month][sdr_name]):
                    db[year][month][sdr_name].append(call_entry)

    with open(CONSOLIDATED_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    
    print("\n✅ SUCESSO: Banco de dados consolidado atualizado!")

if __name__ == "__main__":
    process_all_calls()
