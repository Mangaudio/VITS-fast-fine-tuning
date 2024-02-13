import fastapi
import logging
from model_service import update_model_list, get_tts_fn_by_name, list_model_names
import scipy.io.wavfile as wavf
from fastapi import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import os

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

uvicorn_log = logging.getLogger("uvicorn.access")
uvicorn_log.setLevel(logging.DEBUG)

app = fastapi.FastAPI()


@app.get("/api/list")
def list_models():
    """
    read from models.json and return the list of models
    example:
    [
        {
            "name": "model1",
            "description": "model1 description",
            "path": "/path/to/model1/directory"
        }
    ]
    """
    update_model_list()
    return {"models": list_model_names()}


MAX_TEXT_LEN = 100
LANGUAGE = "日本語"
SPEED = 1.0
SOUND_DIR = "sounds"


class TTSRequest(BaseModel):
    text: str
    model_name: str


@app.post("/api/tts")
def tts(request: TTSRequest):
    """
    text to speech, returns the raw
    """
    update_model_list()
    text = request.text
    model_name = request.model_name
    if len(text) > MAX_TEXT_LEN:
        logger.error("text too long")
        return {"error": f"文本太长，最多{MAX_TEXT_LEN}个字符"}
    language = "日本語"
    ret = get_tts_fn_by_name(model_name)
    if ret is None:
        logger.error(f"model {model_name} not found")
        return {"error": f"模型 {model_name} 未找到"}
    tts_fn, speaker = ret
    msg, (sampling_rate, audio_np) = tts_fn(text, speaker, language, 1.0)
    logger.debug(f"tts result: {msg}, sampling_rate: {sampling_rate}")
    # convert the numpy array to wav file and return it
    if msg != "Success":
        logger.error(f"tts failed: {msg}")
        return {"error": f"tts失败: {msg}"}

    # create a temporary file to store the audio
    filename = f"{uuid.uuid4()}.wav"
    sound_file = os.path.join(SOUND_DIR, filename)
    wavf.write(sound_file, sampling_rate, audio_np)

    return {"file": f"/static/sounds/{filename}"}


app.mount("/static/sounds", StaticFiles(directory="sounds"), name="static sound files")
