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
    """音频转录器，使用Faster-Whisper进行语音转文字"""
    
    def __init__(self, model_size: str = "base"):
        """
        初始化转录器
        
        Args:
            model_size: Whisper模型大小 (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        self.last_detected_language = None
        self.device = os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self.isolate_gpu = os.getenv("WHISPER_ISOLATE_GPU", "true").lower() not in {"0", "false", "no"}
        self._transcribe_lock = asyncio.Lock()
        
    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            logger.info(
                f"正在加载Whisper模型: {self.model_size} "
                f"(device={self.device}, compute_type={self.compute_type})"
            )
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info("模型加载完成")
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                raise Exception(f"模型加载失败: {str(e)}")

    def unload_model(self):
        """释放Whisper模型，避免任务结束后继续占用GPU显存。"""
        if self.model is None:
            return

        logger.info("正在卸载Whisper模型并释放显存")
        model = self.model
        self.model = None
        try:
            inner_model = getattr(model, "model", None)
            unload = getattr(inner_model, "unload_model", None)
            if callable(unload):
                unload()
        except Exception as e:
            logger.warning(f"卸载Whisper内部模型时出错: {str(e)}")
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
            logger.warning(f"清理CUDA缓存时出错: {str(e)}")

    def _should_isolate_gpu(self) -> bool:
        device = (self.device or "").lower()
        return self.isolate_gpu and device != "cpu"

    def _transcribe_in_isolated_process(self, audio_path: str, language: Optional[str]) -> str:
        logger.info(
            f"正在隔离进程中加载Whisper模型: {self.model_size} "
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
                    raise Exception(f"Whisper子进程退出但未返回结果，退出码: {process.exitcode}")

            if not result.get("ok"):
                logger.error(f"Whisper子进程转录失败:\n{result.get('traceback', '')}")
                raise Exception(result.get("error") or "Whisper子进程转录失败")

            self.last_detected_language = result["detected_language"]
            logger.info(f"检测到的语言: {result['detected_language']}")
            logger.info(f"语言检测概率: {result['language_probability']:.2f}")
            logger.info("转录完成，Whisper子进程已退出并释放GPU上下文")
            return result["transcript_text"]
        finally:
            if process.is_alive():
                process.terminate()
            process.join()
            result_queue.close()
            result_queue.join_thread()
    
    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            language: 指定语言（可选，如果不指定则自动检测）
            
        Returns:
            转录文本（Markdown格式）
        """
        async with self._transcribe_lock:
            try:
                # 检查文件是否存在
                if not os.path.exists(audio_path):
                    raise Exception(f"音频文件不存在: {audio_path}")

                if self._should_isolate_gpu():
                    logger.info(f"开始转录音频: {audio_path}")
                    return await asyncio.to_thread(
                        self._transcribe_in_isolated_process,
                        audio_path,
                        language,
                    )
                
                # 加载模型
                self._load_model()
                
                logger.info(f"开始转录音频: {audio_path}")
                
                # 直接调用会阻塞事件循环；放入线程避免阻塞
                def _do_transcribe():
                    return self.model.transcribe(
                        audio_path,
                        language=language,
                        beam_size=5,
                        best_of=5,
                        temperature=[0.0, 0.2, 0.4],  # 使用温度递增策略
                        # 更稳健：开启VAD与阈值，降低静音/噪音导致的重复
                        vad_filter=True,
                        vad_parameters={
                            "min_silence_duration_ms": 900,  # 静音检测时长
                            "speech_pad_ms": 300  # 语音填充
                        },
                        no_speech_threshold=0.7,  # 无语音阈值
                        compression_ratio_threshold=2.3,  # 压缩比阈值，检测重复
                        log_prob_threshold=-1.0,  # 日志概率阈值
                        # 避免错误累积导致的连环重复
                        condition_on_previous_text=False
                    )
                segments, info = await asyncio.to_thread(_do_transcribe)
                
                detected_language = info.language
                self.last_detected_language = detected_language  # 保存检测到的语言
                logger.info(f"检测到的语言: {detected_language}")
                logger.info(f"语言检测概率: {info.language_probability:.2f}")
                
                transcript_text = _build_transcript_text(
                    segments,
                    detected_language,
                    info.language_probability,
                )
                logger.info("转录完成")
                
                return transcript_text
                
            except Exception as e:
                logger.error(f"转录失败: {str(e)}")
                raise Exception(f"转录失败: {str(e)}")
            finally:
                self.unload_model()
    
    def _format_time(self, seconds: float) -> str:
        """
        将秒数转换为时分秒格式
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间字符串
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
        获取支持的语言列表
        """
        return [
            "zh", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no"
        ]
    
    def get_detected_language(self, transcript_text: Optional[str] = None) -> Optional[str]:
        """
        获取检测到的语言
        
        Args:
            transcript_text: 转录文本（可选，用于从文本中提取语言信息）
            
        Returns:
            检测到的语言代码
        """
        # 如果有保存的语言，直接返回
        if self.last_detected_language:
            return self.last_detected_language
        
        # 如果提供了转录文本，尝试从中提取语言信息
        if transcript_text and "**Detected Language:**" in transcript_text:
            lines = transcript_text.split('\n')
            for line in lines:
                if "**Detected Language:**" in line:
                    lang = line.split(":")[-1].strip()
                    return lang if lang else None
        
        return None
