# ASR-service for transcription and diarization of meetings audio recordings
## 📣 Introduction
The purpose of the service is to provide a transcript and summary of audio recordings from meetings. The transcript will be broken down into semantic parts, with time stamps and speaker information, and abstracts for each part. There is also a web-service which can be used to test and demonstrate how the whole ASR-system works
## ⚙ Technology stack
[OpenAI Whisper models](https://github.com/openai/whisper) for transcription
[Pyannote model](https://github.com/pyannote/pyannote-audio) for diarization
sberbank-ai/ruRoberta-large model for word embeddings
[cointegrated/rut5-base-absum model](https://huggingface.co/cointegrated/rut5-base-absum) for summarization
Flask framework for web-interface
## 🛠 Installation
<u>Whole project was developed and tested with Python 3.10</u>
1. Clone repository: 
`git clone https://github.com/DefinitelyNik/ASR-service.git`
2. Install Whisper:
`pip install git+https://github.com/openai/whisper.git` or visit their [repo](https://github.com/openai/whisper) and follow instruction there
3. Install Pyannote diarization model from their [repo](https://github.com/pyannote/pyannote-audio) or [hugginface page](https://huggingface.co/pyannote/speaker-diarization-3.1)
4. Install sberbank-ai/ruRoberta-large model(no huggingface page at the moment, so you can try to use another word embedding model [for example](https://huggingface.co/ai-forever/ruRoberta-large))
5. Install cointegrated/rut5-base-absum model from [huggingface page](https://huggingface.co/cointegrated/rut5-base-absum)
6. Install PyTorch from their [website](https://docs.pytorch.org/get-started/locally/) (tested on cuda 11.8 but should work completely fine on other versions)
7. Install other dependencies:
```
pip install Flask librosa numpy dotenv matplotlib scikit-learn transformers
```
