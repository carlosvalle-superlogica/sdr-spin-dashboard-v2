import os
import csv
import json
import urllib.request
from urllib.error import URLError, HTTPError
from datetime import datetime
import google.generativeai as genai

# ==========================================
# 1. CONFIGURAÇÕES E VARIÁVEIS DE AMBIENTE
# ==========================================
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise ValueError("ERRO CRÍTICO: GEMINI_API_KEY não encontrada nos Secrets do GitHub!")

genai.configure(api_key=GEMINI_KEY)

CSV_FILE = "dados_chamadas.csv"
PROMPT_FILE = "evaluation_prompt.txt"
ANALYSES_DIR = "analises_salvas"
CONSOLIDATED_FILE = "consolidated_data.json"

# ==========================================
# 2. FUNÇÕES AUXILIARES
# ==========================================
def load_prompt():
    """Carrega as regras de negócio da IA."""
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()

def ensure_directories():
    """Garante que as pastas necessárias existam para evitar erros de I/O."""
    os.makedirs(ANALYSES_DIR, exist_ok=True)

def parse_date_to_year_month(date_str):
    """Converte a string de data do HubSpot para Ano e Mês em português."""
    try:
        dt = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
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
    """Remove marcações Markdown residuais caso a IA as inclua."""
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

    if not os.path.exists(CSV_FILE):
        print(f"Aviso: O arquivo '{CSV_FILE}' não foi encontrado. Encerrando processo.")
        return

    prompt_content = load_prompt()
    
    # Carrega ou inicializa a base de dados consolidada
    if os.path.exists(CONSOLIDATED_FILE):
        with open(CONSOLIDATED_FILE, 'r', encoding='utf-8') as f:
            db = json.load(f)
    else:
        db = {}

    # Utiliza 'utf-8-sig' para lidar nativamente com exportações BOM do Excel/HubSpot
    with open(CSV_FILE, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            call_id = row.get("ID do objeto")
            sdr_name = row.get("Atividade atribuída a")
            date_str = row.get("Data da atividade")
            audio_url = row.get("URL de gravação")
            result = row.get("Resultado da chamada")

            # Filtro rigoroso: Ignora chamadas sem áudio ou não atendidas
            if not call_id or not audio_url or result != "Ligação atendida":
                continue

            year, month = parse_date_to_year_month(date_str)
            if not year or not month:
                continue

            saved_analysis_path = os.path.join(ANALYSES_DIR, f"{call_id}.json")
            analysis_data = None

            # [ LÓGICA ANTI-DESPERDÍCIO E CACHE ]
            if os.path.exists(saved_analysis_path):
                print(f"-> [CACHE] Ligação {call_id} já analisada. Puxando histórico...")
                try:
                    with open(saved_analysis_path, 'r', encoding='utf-8') as sf:
                        analysis_data = json.load(sf)
                except json.JSONDecodeError:
                    print(f"Erro ao ler cache da ligação {call_id}. Arquivo corrompido.")
                    continue
            else:
                print(f"-> [NOVA] Analisando ID {call_id} | SDR: {sdr_name} | Data: {date_str}...")
                
                try:
                    # Download blindado do áudio
                    req = urllib.request.Request(
                        audio_url, 
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    )
                    with urllib.request.urlopen(req, timeout=45) as response:
                        audio_bytes = response.read()

                    # Inicializa a IA forçando a saída em formato JSON
                    model = genai.GenerativeModel(
                        "gemini-1.5-flash",
                        generation_config={"response_mime_type": "application/json"}
                    )
                    
                    response_ai = model.generate_content([
                        {
                            "mime_type": "audio/wav", # Aceita outros formatos compatíveis
                            "data": audio_bytes
                        },
                        prompt_content
                    ])

                    # Limpeza e validação do JSON retornado
                    clean_text = clean_json_response(response_ai.text)
                    analysis_data = json.loads(clean_text)
                    
                    # Salva backup individual
                    with open(saved_analysis_path, 'w', encoding='utf-8') as sf:
                        json.dump(analysis_data, sf, ensure_ascii=False, indent=2)

                except (URLError, HTTPError) as e:
                    print(f"Erro de rede ao baixar áudio do ID {call_id}: {e}")
                    continue
                except json.JSONDecodeError as e:
                    print(f"Erro: A IA não retornou um JSON válido para a chamada {call_id}: {e}")
                    continue
                except Exception as e:
                    print(f"Erro inesperado na chamada {call_id}: {e}")
                    continue

            # [ CORREÇÃO CRÍTICA: ESTRUTURA DA ÁRVORE DE DADOS (ANO > MÊS > SDR) ]
            if analysis_data:
                if year not in db:
                    db[year] = {}
                if month not in db[year]:
                    db[year][month] = {}
                if sdr_name not in db[year][month]:
                    db[year][month][sdr_name] = []

                call_entry = {
                    "id_registro": call_id,
                    "data_atividade": date_str,
                    "titulo": row.get("Título da chamada", ""),
                    "duracao": row.get("Duração da chamada (HH:mm:ss)", ""),
                    "deal_associated": row.get("Associated Deal", ""),
                    "analise": analysis_data
                }
                
                # Validação para não duplicar dados no array caso o script re-execute
                if not any(item["id_registro"] == call_id for item in db[year][month][sdr_name]):
                    db[year][month][sdr_name].append(call_entry)

    # ==========================================
    # 4. SALVAMENTO FINAL DO BANCO CONSOLIDADO
    # ==========================================
    with open(CONSOLIDATED_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    
    print("\n✅ SUCESSO: Banco de dados consolidado atualizado com perfeição!")

if __name__ == "__main__":
    process_all_calls()
