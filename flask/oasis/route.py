import io
import json
from oasis import app
from flask import send_file, render_template
from flask import Flask, jsonify, request, render_template

import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from oasis.unicode_jamo import join_jamos


tokenizer = Wav2Vec2Processor.from_pretrained(
    "processor_save")
model = Wav2Vec2ForCTC.from_pretrained(
    "checkpoint")


@app.route("/", endpoint="index")
def index():
    return render_template("index.html")


@app.route("/asr", methods=["POST"])
def asr():
    if request.method == "POST":
        print("start")
        print(request.files)
        file = request.files['audio_data']
        # audio_bytes = file.read()
        audio_input, sample_rate = sf.read(file)
        audio_input = librosa.resample(audio_input.T, sample_rate, 16000)

        input_values = tokenizer(
            audio_input, return_tensors="pt", sampling_rate=16000).input_values

        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]

        transcription = join_jamos(transcription)

        return jsonify({"transcription": transcription})

        # return jsonify({"class_name": "TEST"})


@app.route("/finger", methods=["GET"])
def finger_get():
    return send_file("static/img/finger.jpg", mimetype='image/jpg')
