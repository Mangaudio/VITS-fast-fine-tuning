import json
import os
import hashlib
import logging

from inference import load_model

logger = logging.getLogger(__name__)

models = []


"""
model_cache = {
    "model_path": {
        "md5": "md5 hash of the model G_latest.pth file",
        "counter": 0, // how many times the model has been used, every 20 times, check if the model has been updated
        "tts_func": tts_func,
        "speaker": speaker
    }
}
"""
model_cache = {}


def list_model_names() -> list[str]:
    return [model["name"] for model in models]


def update_model_list():
    global models
    if not os.path.exists("models.json"):
        logging.error("models.json not found")
        return
    with open("models.json", "r") as f:
        models = json.load(f)
        logging.debug(f"models: {models}")


update_model_list()

RELOAD_COUNTER = 20


def load_model_if_changed(model_path) -> bool:
    global model_cache
    model_g = os.path.join(model_path, "OUTPUT_MODEL", "G_latest.pth")
    if not os.path.exists(model_g):
        logging.error(f"Model G_latest.pth not found: {model_g}")
        return False
    need_reload = model_path not in model_cache
    md5 = None
    if model_path in model_cache:
        counter = model_cache[model_path]["counter"]
        old_md5 = model_cache[model_path]["md5"]
        if counter > 0 and counter % RELOAD_COUNTER == 0:
            logger.debug(
                f"Counter reached {RELOAD_COUNTER}, checking if model has been updated"
            )
            with open(model_g, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
                need_reload = need_reload or md5 != old_md5
    if need_reload:
        logger.info(f"Loaded model from {model_path}")
        tts_fn, speakers = load_model(model_path)
        logger.info(f"Loaded speakers in model {model_path}: {speakers}")
        if len(speakers) == 0:
            logger.error(f"No speakers found in model {model_path}")
            return False
        if md5 is None:
            with open(model_g, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
        model_cache[model_path] = {
            "md5": md5,
            "tts_func": tts_fn,
            "speaker": speakers[0],
            "counter": 0,
        }
        return True
    return False


def get_tts_fn_by_name(model_name) -> tuple[any, str] or None:
    for model in models:
        if model["name"] == model_name:
            load_model_if_changed(model["path"])
            # return tts function and speaker name, increment the counter
            model = model_cache[model["path"]]
            model["counter"] += 1
            return (model["tts_func"], model["speaker"])
    return None
