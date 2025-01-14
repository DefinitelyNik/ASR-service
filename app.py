from flask import Flask, render_template, request, redirect, url_for
from model import run_model


app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    transcript = ""
    transcription_time = 0
    transcription_ratio = 0
    diarization_time = 0
    diarization_ratio = 0
    audio_duration = 0
    sum_chunks = ""
    sum_time = 0

    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            content = file.read()
            spk_count = request.form["speakerCount"]
            lang = request.form["lang"]
            model = request.form["whisperModels"]
            print(request.form)
            result = run_model(content=content, model_name=model, language=lang, speaker_count=spk_count)
            transcript = result[0]
            transcription_time = round(result[1],2)
            transcription_ratio = round(result[2],2)
            diarization_time = round(result[3],2)
            diarization_ratio = round(result[4],2)
            audio_duration = round(result[5],2)
            sum_chunks = result[6]
            sum_time = round(result[7],2)

    return render_template("index.html",
                           transcript=transcript,
                           transcription_time=transcription_time,
                           transcription_ratio=transcription_ratio,
                           diarization_time = diarization_time,
                           diarization_ratio = diarization_ratio,
                           audio_duration=audio_duration,
                           summarization=sum_chunks,
                           sum_time=sum_time)


if __name__ == '__main__':
    app.run(debug=False)
