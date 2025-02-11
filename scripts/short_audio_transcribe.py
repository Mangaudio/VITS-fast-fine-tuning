import whisper
import os
import json
import torchaudio
import argparse
import torch
import logging

logger = logging.getLogger(__name__)

lang2token = {
    "zh": "[ZH]",
    "ja": "[JA]",
    "en": "[EN]",
}


def transcribe_one(audio_path, n_mels=80):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    (f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    logger.info(result.text)
    return lang, result.text


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-8s %(name)-12s]  %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJE")
    parser.add_argument("--whisper_size", default="medium")
    parser.add_argument(
        "--process_only",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    if args.languages == "CJE":
        lang2token = {
            "zh": "[ZH]",
            "ja": "[JA]",
            "en": "[EN]",
        }
    elif args.languages == "CJ":
        lang2token = {
            "zh": "[ZH]",
            "ja": "[JA]",
        }
    elif args.languages == "C":
        lang2token = {
            "zh": "[ZH]",
        }
    assert torch.cuda.is_available(), "Please enable GPU in order to run Whisper!"
    model = whisper.load_model(args.whisper_size)
    parent_dir = "./custom_character_voice/"
    speaker_names = list(os.walk(parent_dir))[0][1]
    speaker_annos = []
    total_files = sum([len(files) for r, d, files in os.walk(parent_dir)])
    # resample audios
    # 2023/4/21: Get the target sampling rate
    with open("./configs/finetune_speaker.json", "r", encoding="utf-8") as f:
        hps = json.load(f)
    target_sr = hps["data"]["sampling_rate"]
    processed_files = 0
    for speaker in speaker_names:
        for i, wavfile in enumerate(list(os.walk(parent_dir + speaker))[0][2]):
            # try to load file as audio
            if wavfile.startswith("processed_"):
                continue
            try:
                wav, sr = torchaudio.load(
                    parent_dir + speaker + "/" + wavfile,
                    frame_offset=0,
                    num_frames=-1,
                    normalize=True,
                    channels_first=True,
                )
                wav = wav.mean(dim=0).unsqueeze(0)
                if sr != target_sr:
                    wav = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=target_sr
                    )(wav)
                if wav.shape[1] / sr > 20:
                    logger.warning(f"{wavfile} too long, ignoring\n")
                save_path = parent_dir + speaker + "/" + f"processed_{i}.wav"
                torchaudio.save(save_path, wav, target_sr, channels_first=True)
                if args.process_only:
                    continue
                # transcribe text
                lang, text = transcribe_one(
                    save_path, 128 if args.whisper_size == "large-v3" else 80
                )
                if lang not in list(lang2token.keys()):
                    logger.warning(f"{lang} not supported, ignoring\n")
                    continue
                text = lang2token[lang] + text + lang2token[lang] + "\n"
                speaker_annos.append(save_path + "|" + speaker + "|" + text)

                processed_files += 1
                logger.info(f"Processed: {processed_files}/{total_files}")
            except Exception as e:
                logger.error(f"{wavfile} failed to load as audio: {e}, ignoring\n")
                continue

    # # clean annotation
    # import argparse
    # import text
    # from utils import load_filepaths_and_text
    # for i, line in enumerate(speaker_annos):
    #     path, sid, txt = line.split("|")
    #     cleaned_text = text._clean_text(txt, ["cjke_cleaners2"])
    #     cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
    #     speaker_annos[i] = path + "|" + sid + "|" + cleaned_text
    # write into annotation
    if not args.process_only:
        if len(speaker_annos) == 0:
            logger.warning(
                "No short audios found, this IS expected if you have only uploaded long audios, videos or video links."
            )
            logger.warning(
                "This IS NOT expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported."
            )
        with open("short_character_anno.txt", "w", encoding="utf-8") as f:
            for line in speaker_annos:
                f.write(line)

    # import json
    # # generate new config
    # with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
    #     hps = json.load(f)
    # # modify n_speakers
    # hps['data']["n_speakers"] = 1000 + len(speaker2id)
    # # add speaker names
    # for speaker in speaker_names:
    #     hps['speakers'][speaker] = speaker2id[speaker]
    # # save modified config
    # with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
    #     json.dump(hps, f, indent=2)
    # print("finished")
