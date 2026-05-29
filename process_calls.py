import os
import csv
import json
import time
import urllib.request
import io
import traceback
from urllib.error import URLError
from groq import Groq

# ==========================================
# 1. CONFIGURAÇÕES E VARIÁVEIS DE AMBIENTE
# ==========================================
GROQ_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    raise ValueError("ERRO CRÍTICO: GROQ_API_KEY não encontrada nos Secrets do GitHub!")

client = Groq(api_key=GROQ_KEY)

CSV_FILE = "dados_chamadas.csv"
PROMPT_FILE = "evaluation_prompt.txt"
CONSOLIDATED_FILE = "consolidated_data.json"

# ==========================================
# 2. FUNÇÕES AUXILIARES
# ==========================================
def load_prompt():
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Aviso: Arquivo de prompt não encontrado. Usando prompt padrão. Erro: {e}")
        return "Atue como auditor de SPIN Selling e avalie esta transcrição."

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
# 3. NÚCLEO DE PROCESSAMENTO BLINDADO
# ==========================================
def process_all_calls():
    if not os.path.exists(CSV_FILE):
        print(f"Erro: Arquivo {CSV_FILE} não encontrado!")
        return

    print(f"✅ Arquivo de dados encontrado: {CSV_FILE}")
    prompt_content = load_prompt()
    
    db = {}
    if os.path.exists(CONSOLIDATED_FILE):
        try:
            with open(CONSOLIDATED_FILE, 'r', encoding='utf-8') as f:
                db = json.load(f)
        except json.JSONDecodeError:
            print("Aviso: O arquivo JSON estava vazio ou inválido. Iniciando um novo.")
            db = {}

    with open(CSV_FILE, mode='r', encoding='utf-8-sig') as f:
        # DETECÇÃO AUTOMÁTICA DE DELIMITADOR (VÍRGULA OU PONTO E VÍRGULA)
        sample = f.read(2048)
        delimiter = ';' if ';' in sample else ','
        f.seek(0)
        
        reader = csv.DictReader(f, delimiter=delimiter)
        linhas_processadas = 0
        
        for row in reader:
            call_id = row.get("ID do objeto")
            sdr_name = row.get("Atividade atribuída a") or "SDR Não Identificado"
            date_str = row.get("Data da atividade") or "Data Indisponível"
            audio_url = row.get("URL de gravação")
            result = row.get("Resultado da chamada")
            duration = row.get("Duração da chamada (HH:mm:ss)") or "00:00"
            title = row.get("Título da chamada") or "Chamada de Vendas"

            if not call_id or not audio_url or not audio_url.startswith("http"):
                continue
            
            if result not in ["Ligação atendida", "Connected", "Atendida"]:
                continue

            if call_id in db:
                continue

            print(f"-> [NOVA] Analisando ID {call_id} | SDR: {sdr_name}...")
            
            try:
                req = urllib.request.Request(
                    audio_url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    audio_bytes = response.read()

                transcription = client.audio.transcriptions.create(
                    file=("audio.mp3", io.BytesIO(audio_bytes)),
                    model="whisper-large-v3",
                    response_format="json"
                )
                texto_ligacao = transcription.text

                if len(texto_ligacao.strip()) < 10:
                    print(f"Aviso: Áudio {call_id} sem fala detectada. Ignorando.")
                    continue

                prompt_sistema = (
                    f"{prompt_content}\n\n"
                    "REGRA CRÍTICA DO SISTEMA: Responda APENAS com um objeto JSON válido. "
                    "Use exatamente esta estrutura: "
                    "{\"nota_spin\": 8.5, \"avaliacao\": \"texto\", \"sugestoes\": \"texto\"}"
                )

                chat_completion = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": f"Transcrição:\n\n{texto_ligacao}"}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )

                clean_text = clean_json_response(chat_completion.choices[0].message.content)
                analysis_data = json.loads(clean_text)
                
                try:
                    nota = float(analysis_data.get("nota_spin", 0))
                except (ValueError, TypeError):
                    nota = 0.0

                db[call_id] = {
                    "id": call_id,
                    "sdr": sdr_name,
                    "data": date_str,
                    "titulo": title,
                    "duracao": duration,
                    "audio_url": audio_url,
                    "nota_spin": nota,
                    "avaliacao": analysis_data.get("avaliacao", "Análise sem detalhes."),
                    "sugestoes": analysis_data.get("sugestoes", "Sem sugestões."),
                    "transcricao": texto_ligacao
                }
                
                linhas_processadas += 1
                
                with open(CONSOLIDATED_FILE, 'w', encoding='utf-8') as sf:
                    json.dump(db, sf, ensure_ascii=False, indent=4)

                print(f"✅ Chamada {call_id} analisada (Nota: {nota})")
                time.sleep(3)

            except json.JSONDecodeError:
                print(f"Erro: Falha ao decodificar JSON da IA para o ID {call_id}.")
                time.sleep(3)
                continue
            except URLError as e:
                print(f"Erro de Rede: Falha ao baixar áudio do HubSpot para ID {call_id}. {e}")
                continue
            except Exception as e:
                print(f"Erro inesperado no ID {call_id}: {e}")
                time.sleep(3)
                continue

    print(f"\n✅ SUCESSO: {linhas_processadas} chamadas consolidadas com segurança!")

if __name__ == "__main__":
    try:
        process_all_calls()
    except Exception as erro_critico:
        print("\n❌ ERRO CRÍTICO NO FLUXO PRINCIPAL:")
        traceback.print_exc()
        exit(1)
