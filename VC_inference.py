from logging_conf import suppress_warnings

suppress_warnings()
import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import gradio as gr
import librosa

from text import text_to_sequence, _clean_text

device = "cpu"
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)-8s %(name)-12s]  %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("vc_inference")
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ["日本語", "简体中文", "English", "Mix"]


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(
        text, hps.symbols, [] if is_symbol else hps.data.text_cleaners
    )
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed, enable_random, seed_value):
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
                    enable_random=(enable_random, seed_value),
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn


def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, record_audio, upload_audio):
        input_audio = record_audio if record_audio is not None else upload_audio
        if input_audio is None:
            return "You need to record or upload an audio", None
        sampling_rate, audio = input_audio
        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(
                audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate
            )
        with no_grad():
            y = torch.FloatTensor(audio)
            y = y / max(-y.min(), y.max()) / 0.99
            y = y.to(device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            ).to(device)
            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)
            sid_tgt = LongTensor([target_speaker_id]).to(device)
            audio = (
                model.voice_conversion(
                    spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
        del y, spec, spec_lengths, sid_src, sid_tgt
        return "Success", (hps.data.sampling_rate, audio)

    return vc_fn


tts_fn = None
vc_fn = None
speakers = None

model_cache = {}


def load_model(model_path):
    # get character name by folder name
    character_name = os.path.basename(model_path)
    if character_name in model_cache:
        logger.info(f"Loading model {character_name} from cache")
        return model_cache[character_name]
    logger.info(f"Loading model {character_name} from {model_path}")
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
    vc_fn = create_vc_fn(net_g, hps, speaker_ids)
    model_cache[character_name] = (tts_fn, vc_fn, speakers)
    return tts_fn, vc_fn, speakers


def update_model(model_path):
    global tts_fn, vc_fn, speakers
    tts_fn, vc_fn, speakers = load_model(model_path)


if __name__ == "__main__":
    logger.info(f"Using device: {device}")
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--model_dir",
    #     default="./G_latest.pth",
    #     help="directory to your fine-tuned model",
    # )
    # parser.add_argument(
    #     "--config_dir",
    #     default="./finetune_speaker.json",
    #     help="directory to your model config file",
    # )
    parser.add_argument(
        "--model_dir",
        default="./characters",
        help="directory to your fine-tuned models",
    )

    parser.add_argument(
        "--share", default=False, help="make link public (used in colab)"
    )

    args = parser.parse_args()
    model_dir = args.model_dir
    characters = os.listdir(model_dir)
    # list all models
    logger.info(f"Available models: {characters}")
    if len(characters) == 0:
        raise Exception("No model found!")
    update_model(os.path.join(model_dir, characters[0]))
    app = gr.Blocks()
    with app:
        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(
                        label="Text",
                        placeholder="Type your sentence here",
                        value="こんにちわ。",
                        elem_id=f"tts-input",
                    )
                    model_dropdown = gr.Dropdown(
                        choices=characters, value=characters[0], label="模型 Model"
                    )

                    # select character
                    char_dropdown = gr.Dropdown(
                        choices=speakers, value=speakers[0], label="说话人 Speaker"
                    )

                    def update_model_dropdown(character):
                        update_model(os.path.join(model_dir, character))
                        return gr.Dropdown(
                            choices=speakers, value=speakers[0], label="说话人 Speaker"
                        )

                    model_dropdown.change(
                        update_model_dropdown,
                        inputs=[model_dropdown],
                        outputs=[char_dropdown],
                    )
                    language_dropdown = gr.Dropdown(
                        choices=lang, value=lang[0], label="语言 Language"
                    )
                    duration_slider = gr.Slider(
                        minimum=0.1, maximum=5, value=1, step=0.1, label="速度 Speed"
                    )
                    enable_random = gr.Checkbox(
                        value=True,
                        label="启用随机性 Enable Randomness",
                        # description="启用随机性会导致同样的输入下，不同次的生成结果不同 Enable Randomness will cause different results for the same input",
                    )
                    seed_value = gr.Number(
                        value=0, label="手动随机种子 Manual Random Seed"
                    )
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")

                    def tts_fn_fwd(
                        text, speaker, language, speed, enable_random, seed_value
                    ):
                        return tts_fn(
                            text, speaker, language, speed, enable_random, seed_value
                        )

                    btn.click(
                        tts_fn_fwd,
                        inputs=[
                            textbox,
                            char_dropdown,
                            language_dropdown,
                            duration_slider,
                            enable_random,
                            seed_value,
                        ],
                        outputs=[text_output, audio_output],
                    )
        with gr.Tab("Voice Conversion"):
            gr.Markdown(
                """
                            录制或上传声音，并选择要转换的音色。
            """
            )
            with gr.Column():
                audio = gr.Audio(
                    label="record your voice", sources=["microphone", "upload"]
                )
                source_speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="source speaker"
                )
                target_speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="target speaker"
                )
            with gr.Column():
                message_box = gr.Textbox(label="Message")
                converted_audio = gr.Audio(label="converted audio")
            btn = gr.Button("Convert!")

            def vc_fn_fwd(original_speaker, target_speaker, record_audio, upload_audio):
                return vc_fn(
                    original_speaker, target_speaker, record_audio, upload_audio
                )

            btn.click(
                vc_fn_fwd,
                inputs=[source_speaker, target_speaker, audio],
                outputs=[message_box, converted_audio],
            )
    app.launch(share=False)
