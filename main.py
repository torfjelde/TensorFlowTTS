import os
import argparse

from datetime import datetime

import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

import spacy
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Process text file and output audio.")

parser.add_argument("input_file")
parser.add_argument(
    "-o", "--output",
    help="The base of the filename for the output.",
    default="."
)

# initialize fastspeech2 model.
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")


# initialize mb_melgan model
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")


# inference
processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")


def text2speech(text):
    input_ids = processor.text_to_sequence(text)

    mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )

    # melgan inference
    audio_before = mb_melgan.inference(mel_before)[0, :, 0]
    audio_after = mb_melgan.inference(mel_after)[0, :, 0]

    return audio_before, audio_after


if __name__ == "__main__":
    args = parser.parse_args()

    path = args.input_file
    assert os.path.isfile(path)

    with open(path, "r") as f:
        text = f.read()

    # Assumes we've run the command:
    # `python -m spacy download en_core_web_sm`
    nlp = spacy.load("en_core_web_sm")

    # Split up the sentences.
    doc = nlp(text)

    # results_before = []
    results_after = []
    for sentence in tqdm(doc.sents):
        try:
            audio_before, audio_after = text2speech(str(sentence))
            # Save the results.
            # results_before.append(audio_before.numpy())
            results_after.append(audio_after.numpy())
        except:
            print(f"Failed to convert the sentence {sentence}; trying with split sentence...")
            for sub_sentence in str(sentence).split(","):
                try:
                    audio_before, audio_after = text2speech(sub_sentence)
                    # Save the results.
                    # results_before.append(audio_before.numpy())
                    results_after.append(audio_after.numpy())
                except:
                    print(f"Failed to convert the sub-sentence {sub_sentence}; giving up...")

    # Save everything to file.
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    fname = os.path.splitext(os.path.basename(args.input_file))[0]
    output_path = os.path.abspath(os.path.expanduser(args.output))
    output_fname = f"{current_date}_{fname}"
    print(
        f"Saving output to {output_path} as {output_fname}.wav...",
        end=""
    )
    # sf.write(
    #     os.path.join(output_path, f'{output_fname}_before.wav'),
    #     np.concatenate(results_before), 22050, "PCM_16"
    # )

    # After pitch-adjustment.
    sf.write(
        os.path.join(output_path, f'{output_fname}.wav'),
        np.concatenate(results_after), 22050, "PCM_16"
    )
    print("OK!")
