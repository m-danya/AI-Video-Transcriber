import os
import gc
import asyncio
import multiprocessing as mp
import queue
import traceback
from faster_whisper import WhisperModel
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


def _format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _build_transcript_text(segments: Any, detected_language: str, language_probability: float) -> str:
    transcript_lines = [
        "# Video Transcription",
        "",
        f"**Detected Language:** {detected_language}",
        f"**Language Probability:** {language_probability:.2f}",
        "",
        "## Transcription Content",
        "",
    ]

    for segment in segments:
        start_time = _format_time(segment.start)
        end_time = _format_time(segment.end)
        text = segment.text.strip()

        transcript_lines.append(f"**[{start_time} - {end_time}]**")
        transcript_lines.append("")
        transcript_lines.append(text)
        transcript_lines.append("")

    return "\n".join(transcript_lines)


def _transcribe_in_subprocess(
    result_queue: mp.Queue,
    model_size: str,
    device: str,
    compute_type: str,
    audio_path: str,
    language: Optional[str],
) -> None:
    model = None
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4],
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 900,
                "speech_pad_ms": 300,
            },
            no_speech_threshold=0.7,
            compression_ratio_threshold=2.3,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
        )
        transcript_text = _build_transcript_text(
            segments,
            info.language,
            info.language_probability,
        )
        result_queue.put({
            "ok": True,
            "transcript_text": transcript_text,
            "detected_language": info.language,
            "language_probability": info.language_probability,
        })
    except Exception as e:
        result_queue.put({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
    finally:
        model = None
        gc.collect()


class Transcriber:
    """Audio transcriber that uses Faster-Whisper for speech-to-text."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        self.last_detected_language = None
        self.device = os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self.isolate_gpu = os.getenv("WHISPER_ISOLATE_GPU", "true").lower() not in {"0", "false", "no"}
        self._transcribe_lock = asyncio.Lock()
        
    def _load_model(self):
        """Load the model lazily."""
        if self.model is None:
            logger.info(
                f"Loading Whisper model: {self.model_size} "
                f"(device={self.device}, compute_type={self.compute_type})"
            )
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info("Model loaded")
            except Exception as e:
                logger.error(f"Model load failed: {str(e)}")
                raise Exception(f"Model load failed: {str(e)}")

    def unload_model(self):
        """Unload the Whisper model so GPU memory is released after each task."""
        if self.model is None:
            return

        logger.info("Unloading Whisper model and releasing GPU memory")
        model = self.model
        self.model = None
        try:
            inner_model = getattr(model, "model", None)
            unload = getattr(inner_model, "unload_model", None)
            if callable(unload):
                unload()
        except Exception as e:
            logger.warning(f"Error while unloading the inner Whisper model: {str(e)}")
        model = None
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error while clearing CUDA cache: {str(e)}")

    def _should_isolate_gpu(self) -> bool:
        device = (self.device or "").lower()
        return self.isolate_gpu and device != "cpu"

    def _transcribe_in_isolated_process(self, audio_path: str, language: Optional[str]) -> str:
        logger.info(
            f"Loading Whisper model in an isolated process: {self.model_size} "
            f"(device={self.device}, compute_type={self.compute_type})"
        )
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_transcribe_in_subprocess,
            args=(
                result_queue,
                self.model_size,
                self.device,
                self.compute_type,
                audio_path,
                language,
            ),
        )
        process.start()

        result = None
        try:
            while process.is_alive():
                try:
                    result = result_queue.get(timeout=0.5)
                    break
                except queue.Empty:
                    continue

            process.join()

            if result is None:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    raise Exception(f"Whisper subprocess exited without returning a result, exit code: {process.exitcode}")

            if not result.get("ok"):
                logger.error(f"Whisper subprocess transcription failed:\n{result.get('traceback', '')}")
                raise Exception(result.get("error") or "Whisper subprocess transcription failed")

            self.last_detected_language = result["detected_language"]
            logger.info(f"Detected language: {result['detected_language']}")
            logger.info(f"Language detection probability: {result['language_probability']:.2f}")
            logger.info("Transcription complete; Whisper subprocess exited and released the GPU context")
            return result["transcript_text"]
        finally:
            if process.is_alive():
                process.terminate()
            process.join()
            result_queue.close()
            result_queue.join_thread()
    
    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: audio file path
            language: optional language; if omitted, Whisper auto-detects it
            
        Returns:
            Transcribed text in Markdown format.
        """
        async with self._transcribe_lock:
            try:
                # Check whether the file exists.
                if not os.path.exists(audio_path):
                    raise Exception(f"Audio file does not exist: {audio_path}")

                if self._should_isolate_gpu():
                    logger.info(f"Starting audio transcription: {audio_path}")
                    return await asyncio.to_thread(
                        self._transcribe_in_isolated_process,
                        audio_path,
                        language,
                    )
                
                # Load model.
                self._load_model()
                
                logger.info(f"Starting audio transcription: {audio_path}")
                
                # Direct calls block the event loop; run in a worker thread instead.
                def _do_transcribe():
                    return self.model.transcribe(
                        audio_path,
                        language=language,
                        beam_size=5,
                        best_of=5,
                        temperature=[0.0, 0.2, 0.4],  # Incremental temperature fallback strategy.
                        # More robust: enable VAD and thresholds to reduce silence/noise repetition.
                        vad_filter=True,
                        vad_parameters={
                            "min_silence_duration_ms": 900,  # Silence detection duration.
                            "speech_pad_ms": 300  # Speech padding.
                        },
                        no_speech_threshold=0.7,  # No-speech threshold.
                        compression_ratio_threshold=2.3,  # Compression ratio threshold for repetition detection.
                        log_prob_threshold=-1.0,  # Log probability threshold.
                        # Avoid compounding errors that can cause repeated output.
                        condition_on_previous_text=False
                    )
                segments, info = await asyncio.to_thread(_do_transcribe)
                
                detected_language = info.language
                self.last_detected_language = detected_language  # Save detected language.
                logger.info(f"Detected language: {detected_language}")
                logger.info(f"Language detection probability: {info.language_probability:.2f}")
                
                transcript_text = _build_transcript_text(
                    segments,
                    detected_language,
                    info.language_probability,
                )
                logger.info("Transcription complete")
                
                return transcript_text
                
            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                raise Exception(f"Transcription failed: {str(e)}")
            finally:
                self.unload_model()
    
    def _format_time(self, seconds: float) -> str:
        """
        Convert seconds to a timestamp string.
        
        Args:
            seconds: number of seconds
            
        Returns:
            Formatted timestamp string.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def get_supported_languages(self) -> list:
        """
        Get supported languages.
        """
        return [
            "zh", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no"
        ]
    
    def get_detected_language(self, transcript_text: Optional[str] = None) -> Optional[str]:
        """
        Get the detected language.
        
        Args:
            transcript_text: optional transcript text used to extract language metadata
            
        Returns:
            Detected language code.
        """
        # Return the saved language if available.
        if self.last_detected_language:
            return self.last_detected_language
        
        # If transcript text was provided, try to extract language metadata from it.
        legacy_detected_language_label = "**\u68c0\u6d4b\u8bed\u8a00:**"
        if transcript_text and (
            "**Detected Language:**" in transcript_text
            or legacy_detected_language_label in transcript_text
        ):
            lines = transcript_text.split('\n')
            for line in lines:
                if "**Detected Language:**" in line or legacy_detected_language_label in line:
                    lang = line.split(":")[-1].strip()
                    return lang if lang else None
        
        return None
