from flask import Flask, request, jsonify
import json
import wave
import pyaudio
import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient
from datetime import datetime
import io
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

app = Flask(__name__)

# 환경 변수 로드
load_dotenv()

# Azure 설정
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

BLOB_AUDIO_CONTAINER = "audio-files"
STT_RESULTS_CONTAINER = "stt-results"
SUMMARIZED_RESULTS_CONTAINER = "sum-results"

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT
)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

@app.route("/record", methods=["POST"])
def record_and_upload():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob_name = f"live_audio_{timestamp}.wav"
    blob_client = blob_service_client.get_blob_client(container=BLOB_AUDIO_CONTAINER, blob=blob_name)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("녹음 시작...")
    frames = []
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("녹음 종료")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wav_data = io.BytesIO()
    with wave.open(wav_data, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    blob_client.upload_blob(wav_data.getvalue(), overwrite=True)
    return jsonify({"message": "녹음 파일 업로드 완료", "blob_name": blob_name})

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    data = request.get_json()
    blob_name = data.get("blob_name")

    blob_client = blob_service_client.get_blob_client(container=BLOB_AUDIO_CONTAINER, blob=blob_name)
    blob_data = blob_client.download_blob().readall()

    speech_key = os.getenv("SPEECH_API_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = "ko-KR"

    stream_reader = io.BytesIO(blob_data)
    audio_input_stream = speechsdk.audio.AudioConfig(filename=blob_name)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input_stream)

    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return jsonify({"stt_text": result.text})
    else:
        return jsonify({"error": "음성 인식 실패"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
