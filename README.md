# SP Crime Predictor API (FastAPI)

API para servir um modelo `modelo.pkl` (scikit-learn) com previsões baseadas em:
- `BAIRRO` (bairros centrais de SP)
- `HORA` (0–23)
- `DIA_SEMANA` (SEGUNDA..DOMINGO, aceita "segunda-feira" etc.)

## Rotas
- `GET /health` — status do serviço e infos do modelo
- `GET /model_info` — detalhes do modelo (nome, tempo de carga, classes, proba)
- `GET /labels` — classes do modelo (se disponíveis)
- `POST /predict` — previsão (single ou batch)

### Exemplo de request
```bash
curl -X POST https://APP.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "items":[
      {"BAIRRO":"REPÚBLICA","HORA":1,"DIA_SEMANA":"SEGUNDA"}
    ]
  }'
```

## Deploy no Render

### Configurações necessárias:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`

### Variáveis de ambiente (opcionais):
- `MODEL_PATH`: caminho para o arquivo do modelo (padrão: "model.pkl")
- `ALLOWED_ORIGINS`: domínios permitidos para CORS (padrão: "*")

### Arquivos necessários:
- `main.py` - aplicação FastAPI
- `model.pkl` - modelo treinado
- `requirements.txt` - dependências Python
