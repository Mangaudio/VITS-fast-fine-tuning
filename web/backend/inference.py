import torch
from torch import no_grad, LongTensor
import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons as commons
from mel_processing import spectrogram_torch
import utils as utils
from models import SynthesizerTrn
import gradio as gr
import librosa
import logging
from text import text_to_sequence, _clean_text

device = "cpu"
logger = logging.getLogger(__name__)


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(
        text, hps.symbols, [] if is_symbol else hps.data.text_cleaners
    )
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}


def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = (
                model.infer(
                    x_tst,
                    x_tst_lengths,
                    sid=sid,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0 / speed,
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn


def load_model(model_path):
    logger.info(f"Loading model from {model_path}")
    model_g = os.path.join(model_path, "OUTPUT_MODEL", "G_latest.pth")
    model_config = os.path.join(model_path, "finetune_speaker.json")
    hps = utils.get_hparams_from_file(model_config)

    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(model_g, net_g, None)
    speaker_ids = hps.speakers
    speakers = list(hps.speakers.keys())
    tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    return tts_fn, speakers
