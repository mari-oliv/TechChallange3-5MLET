# Pós Tech Machine Learning Engineering, turma 5MELT 

## Site de predição: https://sp-crime-predict.lovable.app/

Desenvolvido por alunos da FIAP Pós-Tech Machine Learning Engineering:
* Lucas Barros Cordeiro
* Pedro Costa Mello
* Tarik Vieira Ghazzaoui
* Marina Oliveira Neves
* João Gabriel Andrade



# API de Inferência (FastAPI + Gunicorn)

API minimalista para servir um modelo de Machine Learning (scikit-learn) em produção (Render) sem Docker.

- Rotas:
  - `GET /status` → `{"status":"ok"}`
  - `POST /predict` → recebe JSON `{ "dia_semana: "...", "hora": 13, "bairro": "REPÚBLICA" }` e retorna:
    ```json
    {
      "crime_previsto": "FURTO",
      "top5": [
        {"classe": "FURTO/ROUBO A TRANSEUNTE", "prob": 0.72},
        {"classe": "FURTO/ROUBO DE VEÍCULO", "prob": 0.18},
        {"classe": "ESTELIONATO", "prob": 0.06},
        {"classe": "OUTROS", "prob": 0.03},
        {"classe": "LESÃO", "prob": 0.01}
      ]
    }
    ```

## Estrutura

```
.
├── app
│   ├── main.py            # App FastAPI e rotas
│   └── model_loader.py    # Utilitários: carregamento do modelo e inferência
├── model
│   └── model.pkl          # (opcional) seu modelo; se ausente, API responde 503 no /predict
├── Procfile               # Comando de inicialização para Render
├── requirements.txt
├── .gitignore
└── README.md
```

> **Importante:** O modelo deve ser um `Pipeline` do scikit-learn que aceite um `DataFrame` com colunas **`DATA`**, **`HORA`** e **`BAIRRO`** (ou equivalentes tratados dentro do pipeline).

## Carregamento do Modelo

- Caminho padrão: `./model/model.pkl`
- Você pode sobrescrever com a variável de ambiente **`MODEL_PATH`** (ex.: um caminho absoluto no disco).  
- Caso o arquivo não exista ou falhe ao carregar, a rota `/predict` retorna **503** com `{"detail": "modelo não carregado"}` (a API nunca trava).

## Execução Local

1. **Python 3.10+** recomendado.
2. Crie um ambiente e instale dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. (Opcional) Coloque seu `model.pkl` em `./model/model.pkl` ou defina `MODEL_PATH`:
```bash
export MODEL_PATH=/caminho/absoluto/para/model.pkl
```

4. Rode com Uvicorn em desenvolvimento:
```bash
python -m uvicorn app.main:app --reload
```
A API estará em `http://127.0.0.1:8000` (ou a porta que você especificar com `--port`).

5. Rode com Gunicorn (simulando produção):
```bash
PORT=8000 gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:$PORT app.main:app
```

## Deploy no Render

1. Faça push deste repositório para o Git.
2. No Render, crie um **Web Service** apontando para o repositório.
3. Defina:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: (opcional, Render lê o `Procfile`)  
     `gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:$PORT app.main:app`
   - Variáveis de ambiente:
     - `MODEL_PATH` (opcional) se o modelo não estiver em `./model/model.pkl`.
4. Render define a porta em **`$PORT`**, já respeitada no `Procfile`.

> Se o modelo não estiver presente no container do Render, a rota `/predict` seguirá retornando **503** sem derrubar a API.

## Esquema de Resposta e Fallbacks

- Se o modelo possuir `predict_proba`, a API usa diretamente as probabilidades e retorna as **top-5** classes.
- Caso **não** tenha `predict_proba` mas tenha `decision_function`, aplica **softmax** nos scores (ou uma sigmoide no caso binário) para estimar probabilidades e construir as **top-5**.
- Na ausência de ambos, retorna somente a classe prevista com probabilidade 1.0.

## Validação (Pydantic)

- `POST /predict` exige os campos:
  - `dia_semana` (string)
  - `hora` (inteiro 0–23)
  - `bairro` (string não vazia)

Faltando qualquer campo, o FastAPI retorna **422**.

## Exemplos `curl`

Status:
```bash
curl -s http://localhost:8000/status
```

Predict (local):
```bash
curl -s -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"dia_semana": "segunda","hora":13,"bairro":"REPÚBLICA"}'
```

Predict (Render, supondo serviço em produção):
```bash
curl -s -X POST https://techchallange3-5mlet.onrender.com/predict  -H "Content-Type: application/json"   -d '{"dia_semana": "segunda","hora":13,"bairro":"REPÚBLICA"}'
```

## Notas

- O **gunicorn** está configurado com **`UvicornWorker`** e bind em `0.0.0.0:$PORT`.
- Ajuste o número de workers conforme a necessidade (ex.: `-w 2`), lembrando que alguns provedores têm limites de CPU/memória.
- O diretório `model/` está no `.gitignore` por padrão — inclua seu `model.pkl` no build/deploy ou use `MODEL_PATH`.
