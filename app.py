import json
import torch
import whisperx
import gc
import io
import numpy as np
from scipy.io import wavfile
import requests
import time


device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"


class InferlessPythonModel:
    def initialize(self):
        print("Initializing")
        self.model_name = "NbAiLab/nb-whisper-tiny"
        logger = TimeLogger()
        self.model = whisperx.load_model(
            self.model_name,
            device=device,
            compute_type=compute_type,
        )
        logger.log(f"Loaded x model {self.model_name}")

    def infer(self, inputs):
        audio_url = inputs["audio_url"]

        logger = TimeLogger()
        audio_bytes = download_file(audio_url)
        logger.log("Downloaded audio")
        audio_np, sample_rate, duration = load_wav_from_bytes(audio_bytes)
        logger.log("Loaded audio")
        print(f"Audio duration: {duration:.2f}s")

        asr_result = self.model.transcribe(audio_np)
        logger.log("Inference")

        return {"transcribed_output": json.dumps(asr_result)}

    def finalize(self):
        pass


def download_file(url: str):
    """Download a file from a URL and return the content."""
    response = requests.get(url)
    response.raise_for_status()  # Ensure download was successful
    return response.content


def load_wav_from_bytes(bytes_data: bytes):
    """Load a WAV file from bytes data into a NumPy array and get its duration."""
    with io.BytesIO(bytes_data) as wav_file:
        # scipy.io.wavfile.read might not be the most efficient for large files
        # but it works for demonstration purposes
        sample_rate, audio_data = wavfile.read(wav_file)
        duration = len(audio_data) / sample_rate
        # Normalize audio_data if it's not already in float32 format
        if audio_data.dtype != np.float32:
            # Assuming the buffer is 16-bit (for example, pcm_s16le)
            audio_np = audio_data.astype(np.float32) / 32768
        else:
            audio_np = audio_data
    return audio_np, sample_rate, duration


class TimeLogger:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.labels = {}

    def log(self, label=None):
        """
        Logs the time since the start or since the last log for a given label.

        Parameters:
        - label: str, optional label to associate with the timing. If provided,
                 logs the time under the given label. If not provided, logs the
                 time since instantiation or the last unlabeled log.
        """
        current_time = time.perf_counter()
        if label:
            if label in self.labels:
                elapsed = current_time - self.labels[label]["last_log"]
                total_elapsed = current_time - self.labels[label]["start"]
                print(
                    f"{label}: {elapsed:.2f}s since last log, {total_elapsed:.2f}s total"
                )
            else:
                elapsed = current_time - self.start_time
                self.labels[label] = {"start": current_time, "last_log": current_time}
                print(f"{label}: {elapsed:.2f}s since start")
        else:
            elapsed = current_time - self.start_time
            print(f"Elapsed time: {elapsed:.2f}s")
        if label:
            self.labels[label]["last_log"] = current_time
        return elapsed

    def end(self, label=None):
        """
        Logs the total time since the start or since the first log for a given label.

        Parameters:
        - label: str, optional label to conclude timing for. If provided, logs the
                 total time under the given label. If not provided, logs the total
                 time since instantiation.
        """
        current_time = time.perf_counter()
        if label and label in self.labels:
            total_elapsed = current_time - self.labels[label]["start"]
            print(f"{label}: {total_elapsed:.2f}s total")
        else:
            total_elapsed = current_time - self.start_time
            print(f"Total elapsed time: {total_elapsed:.2f}s")
        return total_elapsed
