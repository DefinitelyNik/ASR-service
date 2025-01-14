import io
import librosa
import torch
import whisperx
import time
from dotenv import load_dotenv
import os
import numpy as np

from text_clusterisation import run_summarization


def run_model(content, model_name = 'large-v3', language = None, speaker_count = 1):

    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')

    speaker_count = check_int(speaker_count)
    if language == "None": language = None
    print(model_name, speaker_count, type(speaker_count))

    batch_size = 16
    compute_type = "bfloat16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)

    buffer = io.BytesIO(content)
    sr = 16000
    wavefile = librosa.load(buffer, sr=sr)
    model_audio = wavefile[0]

    #temp_duration = librosa.get_duration(y=model_audio, sr=sr)
    #samples_10_seconds = int(temp_duration * 0.1 * sr)
    #first_10_seconds = model_audio[:samples_10_seconds]
    #modified_audio = np.concatenate((first_10_seconds, model_audio))

    start_time = time.time()
    result = model.transcribe(model_audio, batch_size=batch_size)
    transcription_time = time.time() - start_time
    audio_duration = librosa.get_duration(y = model_audio, sr=sr)
    transcription_ratio = transcription_time/audio_duration

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    start_time = time.time()
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, model_audio, device,
                            return_char_alignments=False)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
    diarize_segments = diarize_model(model_audio, num_speakers=2)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    diarization_time = time.time() - start_time
    diarization_ratio = diarization_time/audio_duration

    start_end = []
    text = []
    speaker = []

    for item in result['segments']:
        start_end.append(str(item['start']) + " - " + str(item['end']))
        text.append(item["text"])
        try:
            speaker.append(item["speaker"])
        except KeyError:
            speaker.append("NO SPEAKER:")

    buff_dict = {'segment': start_end, 'text': text, 'speaker': speaker}

    transcript = ""
    raw_text = ""
    for i in range(len(buff_dict['segment'])):
        transcript += "[%s] - %s: %s \n" % (buff_dict['segment'][i], buff_dict['speaker'][i], buff_dict['text'][i])
        raw_text += buff_dict['text'][i] + "\n"

    start_time = time.time()
    sum_result = run_summarization(raw_text)
    sum_time = time.time() - start_time

    chunks = ""
    for i in sum_result:
        chunks += i + "\n"

    return transcript, transcription_time, transcription_ratio, diarization_time, diarization_ratio, audio_duration, chunks, sum_time


def check_int(data):
    try:
        num = int(data)
        if num < 1: num = None
        return num
    except TypeError:
        print("Received wrong speaker number(wrong data type)")
    return 1