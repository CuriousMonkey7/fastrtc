import json
import logging
import numpy as np
import torch
from typing import List, Tuple
from dataclasses import dataclass

from huggingface_hub import hf_hub_download
from ..utils import AudioChunk
from .protocol import PauseDetectionModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



@dataclass
class HumAwareVADOptions:
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    window_size_samples: int = 1024
    speech_pad_ms: int = 400


class HumAwareVADModel(PauseDetectionModel):
    @staticmethod
    def download_and_load_model() -> torch.jit.ScriptModule:
        """Download and load the HumAware-VAD model."""
        config_path = hf_hub_download(repo_id="CuriousMonkey7/HumAware-VAD", filename="config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        model_path = hf_hub_download(repo_id="CuriousMonkey7/HumAware-VAD", filename=config["model_file"])
        model = torch.jit.load(model_path, map_location=torch.device("cpu"))
        model.eval()
        return model

    def warmup(self) -> None:
        """Perform a warmup pass to initialize the model."""
        logger.info("Warming up the model...")
        for _ in range(10):
            dummy_audio = np.zeros(102400, dtype=np.float32)
            self.vad((24000, dummy_audio), None)
        logger.info("Warmup complete.")

    def __init__(self) -> None:
        """Initialize the HumAwareVADModel with a pre-trained model."""
        self.model = self.download_and_load_model()

    @staticmethod
    def collect_chunks(audio: np.ndarray, chunks: List[AudioChunk]) -> np.ndarray:
        """Collect speech chunks from the given audio."""
        return np.concatenate([audio[int(chunk["start"]): int(chunk["end"])] for chunk in chunks]) if chunks else np.array([], dtype=np.float32)

    def get_speech_timestamps(
        self, 
        audio: np.ndarray, 
        options: HumAwareVADOptions, 
        sampling_rate: int = 16000
    ) -> List[AudioChunk]:
        """Extract speech timestamps from the given audio.

        Args:
        - audio: One-dimensional float array.
        - options: Options for VAD processing.
        - sampling_rate: The sample rate of the input audio.

        Returns:
        - List of dictionaries containing start and end timestamps of detected speech chunks.
        """
        logger.debug("Processing audio for speech timestamps...")
        if not torch.is_tensor(audio):
            audio = torch.Tensor(audio)
        if len(audio.shape) > 1:
            audio = audio.squeeze()
            if len(audio.shape) > 1:
                raise ValueError("Audio should be a single channel.")

        if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
            step = sampling_rate // 16000
            sampling_rate = 16000
            audio = audio[::step]
        else:
            step = 1

        if sampling_rate not in [8000, 16000]:
            raise ValueError("Only 8000 and 16000 Hz are supported.")

        self.model.reset_states()
        window_size_samples = 512 if sampling_rate == 16000 else 256
        min_speech_samples = sampling_rate * options.min_speech_duration_ms / 1000
        speech_pad_samples = sampling_rate * options.speech_pad_ms / 1000
        audio_length_samples = len(audio)
        speech_probs = [
            self.model(torch.nn.functional.pad(audio[i: i + window_size_samples], (0, max(0, window_size_samples - len(audio[i: i + window_size_samples])))), sampling_rate).item()
            for i in range(0, audio_length_samples, window_size_samples)
        ]

        speeches: List[AudioChunk] = []
        triggered = False
        current_speech: dict[str, int] = {}
        temp_end = 0

        for i, speech_prob in enumerate(speech_probs):
            if speech_prob >= options.threshold:
                if not triggered:
                    triggered = True
                    current_speech["start"] = window_size_samples * i
                temp_end = 0
            elif triggered:
                if not temp_end:
                    temp_end = window_size_samples * i
                if (window_size_samples * i) - temp_end >= sampling_rate * options.min_silence_duration_ms / 1000:
                    current_speech["end"] = temp_end
                    if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                        speeches.append(current_speech)
                    current_speech, triggered, temp_end = {}, False, 0

        if current_speech and (audio_length_samples - current_speech["start"]) > min_speech_samples:
            current_speech["end"] = audio_length_samples
            speeches.append(current_speech)

        if step > 1:
            for speech_dict in speeches:
                speech_dict["start"] *= step
                speech_dict["end"] *= step

        logger.debug("Speech timestamps extraction complete.")
        return speeches

    def vad(
        self, 
        audio: Tuple[int, np.ndarray], 
        options: None | HumAwareVADOptions =None 
    ) -> Tuple[float, List[AudioChunk]]:
        """Perform Voice Activity Detection (VAD) on the given audio.

        Args:
        - audio: A tuple containing the sampling rate and audio array.
        - options: Options for VAD processing.

        Returns:
        - The duration of detected speech in seconds.
        - A list of speech timestamps.
        """
        sampling_rate, audio_ = audio
        if audio_.dtype != np.float32:
            audio_ = audio_.astype(np.float32) / 32768.0
        if sampling_rate != 16000:
            import librosa
            audio_ = librosa.resample(audio_, orig_sr=sampling_rate, target_sr=16000)
        if not options:
            options = HumAwareVADOptions()

        speech_timestamps = self.get_speech_timestamps(audio_, options)
        audio_ = self.collect_chunks(audio_, speech_timestamps)
        duration_after_vad = audio_.shape[0] / 16000  
        return duration_after_vad, speech_timestamps
