import os
from datetime import datetime

import pexpect
import spacy
from tqdm import tqdm

import gradio as gr
import soundfile as sf
import numpy as np

from main import text2speech

output_path = "/home/tor/Dropbox/text-to-speech/outputs/"
nlp = spacy.load("en_core_web_sm")


def text_to_wav(text):
    # Tokenize so we get the sentences.
    doc = nlp(text)

    results = []
    for sentence in tqdm(list(doc.sents)):
        try:
            _, audio = text2speech(str(sentence))
            # Save the results.
            results.append(audio.numpy())
        except Exception:
            print(f"Failed to convert the sentence {sentence}; trying with split sentence...")
            for sub_sentence in str(sentence).split(","):
                try:
                    _, audio = text2speech(sub_sentence)
                    # Save the results.
                    results.append(audio.numpy())
                except Exception:
                    print(f"Failed to convert the sub-sentence {sub_sentence}; giving up...")

    # Save everything to file.
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    # fname = os.path.splitext(os.path.basename(input_file))[0]
    # output_fname = f"{current_date}_{fname}"
    output_fname = current_date
    tmp_path = f"/tmp/{output_fname}.wav"
    print(
        f"Saving output to {tmp_path}...",
        end=""
    )

    # After pitch-adjustment.
    sf.write(
        tmp_path,
        np.concatenate(results), 22050, "PCM_16"
    )
    print("OK!")

    print("Converting to MP3...", end="")
    full_output_path = os.path.join(output_path, f'{output_fname}.mp3')
    pexpect.run(f"ffmpeg -i {tmp_path} -acodec mp3 {full_output_path}")
    print("OK!")

    return full_output_path

demo = gr.Interface(
    fn=text_to_wav,
    inputs=gr.Textbox(lines=10, placeholder="Text here..."),
    outputs=gr.Audio()
)

demo.launch()
