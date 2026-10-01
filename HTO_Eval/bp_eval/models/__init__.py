"""Model registry: the single place train.py looks up a model class by name.

To add a new model: implement BaseBPModel in its own file under
bp_eval/models/, import it here, and add one line to MODEL_REGISTRY.
config['model_type'] in a YAML config then selects it by that key.
"""
from bp_eval.models.base import BaseBPModel
from bp_eval.models.bert_stub import BERTModel
from bp_eval.models.logistic_regression import LogisticRegressionBaseline
from bp_eval.models.lstm import LSTMModel

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionBaseline,
    "lstm": LSTMModel,
    "bert": BERTModel,
}


def build_model(model_type: str, **hyperparams) -> BaseBPModel:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Registered models: "
            f"{list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](**hyperparams)
