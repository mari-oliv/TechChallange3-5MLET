
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import joblib

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(".", "model", "model.pkl"))


def load_model():
    """
    Carrega o modelo do caminho em MODEL_PATH (env) ou ./model/model.pkl.
    Retorna o objeto de modelo ou None se não existir.
    """
    path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    if not os.path.exists(path):
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception:
        # Não propagar exceção para não travar a aplicação
        return None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1)
    if x.ndim > 1:
        x = x.ravel()
    x = x - np.max(x)
    exps = np.exp(x)
    denom = np.sum(exps)
    if denom == 0:  # evita divisão por zero
        return np.ones_like(exps) / exps.size
    return exps / denom


def _build_input_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Normaliza o payload para o DataFrame esperado pelo Pipeline:
    colunas 'DATA', 'HORA', 'BAIRRO'. Mantém valores como strings/nums.
    """
    data = payload.get("data")
    hora = payload.get("hora")
    bairro = payload.get("bairro")

    # Monta DataFrame com nomes em caixa alta conforme suposição do Pipeline
    df = pd.DataFrame([{
        "DATA": data,
        "HORA": hora,
        "BAIRRO": bairro
    }])
    return df


def infer(model: Any, payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, float]]]:
    """
    Executa inferência única.
    Retorna (crime_previsto, top5[{classe, prob}]).
    """
    df = _build_input_df(payload)

    # Predição da classe prevista
    try:
        if hasattr(model, "predict"):
            y_pred = model.predict(df)
            crime_previsto = str(y_pred[0])
        else:
            crime_previsto = ""
    except Exception:
        crime_previsto = ""

    # Probabilidades/top-5
    top5: List[Dict[str, float]] = []
    try:
        classes = getattr(model, "classes_", None)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            if classes is None:
                # Em casos raros sem classes_, deduz do shape
                classes = [str(i) for i in range(len(proba))]
            idx_sorted = np.argsort(proba)[::-1][:5]
            for i in idx_sorted:
                top5.append({"classe": str(classes[i]), "prob": float(proba[i])})

        elif hasattr(model, "decision_function"):
            scores = model.decision_function(df)
            # Normaliza formatos possíveis
            scores = np.asarray(scores)
            if scores.ndim == 0:
                # escalar -> binário (classe positiva), aplica sigmoid-like
                p1 = 1.0 / (1.0 + np.exp(-float(scores)))
                proba = np.array([1 - p1, p1])
                if classes is None:
                    classes = np.array([0, 1])
            elif scores.ndim == 1:
                # vetor por amostra única
                proba = _softmax(scores)
                if classes is None:
                    classes = np.arange(len(proba))
            else:
                # (n_amostras, n_classes) -> pegar a primeira
                proba = _softmax(scores[0])
                if classes is None:
                    classes = np.arange(len(proba))

            proba = np.asarray(proba)
            classes = np.asarray(classes)
            idx_sorted = np.argsort(proba)[::-1][:5]
            for i in idx_sorted:
                top5.append({"classe": str(classes[i]), "prob": float(proba[i])})

        else:
            # Sem proba/decision_function: devolve apenas a classe prevista
            if crime_previsto != "":
                top5 = [{"classe": crime_previsto, "prob": 1.0}]
            else:
                top5 = []

        # Se crime_previsto vazio, usa a melhor do top-1
        if not crime_previsto and top5:
            crime_previsto = str(top5[0]["classe"])

    except Exception:
        # Em caso de falha, retorna apenas a classe prevista (se houver)
        if crime_previsto:
            top5 = [{"classe": crime_previsto, "prob": 1.0}]
        else:
            top5 = []

    return crime_previsto, top5
