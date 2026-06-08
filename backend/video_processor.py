import os
import re
import shutil
import uuid
import asyncio
import subprocess
import yt_dlp
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Video processor that uses yt-dlp to download and convert media."""
    
    def __init__(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',  # Prefer the best available audio source.
            'outtmpl': '%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                # Convert during extraction to mono 16 kHz for compact, stable output.
                'preferredcodec': 'm4a',
                'preferredquality': '192'
            }],
            # Global FFmpeg args: mono + 16 kHz sample rate + faststart.
            'postprocessor_args': ['-ac', '1', '-ar', '16000', '-movflags', '+faststart'],
            'prefer_ffmpeg': True,
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,  # Force single-video downloads instead of playlists.
        }

    async def normalize_local_media_to_m4a(self, input_path: Path, output_dir: Path) -> str:
        """
        Convert a local audio/video upload to mono 16 kHz AAC m4a for Faster-Whisper.
        This matches the yt-dlp postprocessor settings.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]
        out_path = output_dir / f"upload_norm_{unique_id}.m4a"

        cmd = [
            "ffmpeg", "-y", "-nostdin", "-i", str(input_path.resolve()),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(out_path.resolve()),
        ]

        def _run():
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip()
                raise Exception(f"FFmpeg conversion failed: {err[:800]}")
            if not out_path.exists():
                raise Exception("FFmpeg did not generate an output file")

        await asyncio.to_thread(_run)
        return str(out_path)
    
    async def fetch_subtitles(self, url: str, output_dir: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Try platform subtitles first; this is much faster than downloading audio.

        Returns:
            (subtitle_markdown, video_title, language_code)
            subtitle_markdown is None when no usable subtitles are available.
        """
        import asyncio

        output_dir.mkdir(exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]
        sub_dir = output_dir / f"subs_{unique_id}"

        try:
            # 1. Fast probe: get video info and subtitle availability without downloading media.
            check_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
            with yt_dlp.YoutubeDL(check_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, url, False)

            video_title = info.get("title", "unknown")
            manual_subs: dict = info.get("subtitles") or {}
            auto_caps: dict = info.get("automatic_captions") or {}

            # Filter out non-speech tracks such as live_chat.
            manual_langs = [k for k in manual_subs if not k.startswith("live_chat")]
            auto_langs = [k for k in auto_caps if not k.startswith("live_chat")]

            if not manual_langs and not auto_langs:
                logger.info(f"No usable subtitles found for video: {url}")
                return None, video_title, None

            # Prefer manual subtitles, then automatic captions.
            prefer_manual = bool(manual_langs)
            candidate_langs = manual_langs if prefer_manual else auto_langs

            # Choose language by priority: English, Simplified Chinese, Traditional Chinese, then the first available language.
            _priority = ["en", "en-orig", "zh-Hans", "zh-Hant", "zh", "ja", "ko", "fr", "de", "es"]
            prefer_lang = next(
                (lang for lang in _priority if lang in candidate_langs),
                candidate_langs[0],
            )
            logger.info(
                f"Found {'manual' if prefer_manual else 'automatic'} subtitles, selected language: {prefer_lang}"
                f" ({len(candidate_langs)} candidates)"
            )

            # 2. Download subtitles only, skipping audio/video.
            sub_dir.mkdir(exist_ok=True)
            dl_opts = {
                "writesubtitles": prefer_manual,
                "writeautomaticsub": not prefer_manual,
                "subtitlesformat": "vtt/srt/best",
                "subtitleslangs": [prefer_lang],
                "skip_download": True,
                "outtmpl": str(sub_dir / "sub.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [url])

            # 3. Locate the downloaded subtitle file.
            sub_files = list(sub_dir.glob("*.vtt")) + list(sub_dir.glob("*.srt"))
            if not sub_files:
                logger.warning("No subtitle file found after download; falling back to audio mode")
                return None, video_title, None

            sub_file = sub_files[0]

            # Extract language code from the filename (for example, sub.en.vtt -> en).
            stem_parts = sub_file.stem.split(".")
            file_lang = stem_parts[-1] if len(stem_parts) > 1 else prefer_lang

            # 4. Parse the subtitle file.
            if sub_file.suffix == ".vtt":
                entries = self._parse_vtt(str(sub_file))
            else:
                entries = self._parse_srt(str(sub_file))

            if not entries:
                logger.warning("Parsed subtitles are empty; falling back to audio mode")
                return None, video_title, None

            # 5. Format as Markdown compatible with Whisper output.
            formatted = self._format_subtitle_entries(entries, file_lang)
            logger.info(f"Subtitles fetched successfully: lang={file_lang}, entries={len(entries)}")
            return formatted, video_title, file_lang

        except Exception as e:
            logger.warning(f"Subtitle fetch failed; falling back to audio download: {e}")
            return None, None, None
        finally:
            if sub_dir.exists():
                try:
                    shutil.rmtree(str(sub_dir))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Subtitle parsing helpers
    # ------------------------------------------------------------------

    def _parse_vtt(self, filepath: str) -> list:
        """Parse a WebVTT subtitle file and return deduplicated entries.

        Handles YouTube automatic caption "rolling append" cues, where the same
        sentence is spread across multiple incrementally appended cues. Only the
        final version of each group is kept.
        """
        raw_entries = []
        seen_texts: set = set()

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read VTT file: {e}")
            return []

        # Remove the WEBVTT header and split cue blocks by blank lines.
        content = re.sub(r"^WEBVTT[^\n]*\n", "", content)
        blocks = re.split(r"\n{2,}", content.strip())

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")
            timing_idx = next((i for i, l in enumerate(lines) if "-->" in l), -1)
            if timing_idx < 0:
                continue

            timing_line = lines[timing_idx]
            text_lines = lines[timing_idx + 1:]

            match = re.match(
                r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*-->\s*"
                r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)",
                timing_line,
            )
            if not match:
                continue

            start_str = self._normalize_time(match.group(1))
            end_str = self._normalize_time(match.group(2))

            raw_text = " ".join(text_lines)
            # Remove HTML / VTT inline tags, including YouTube word-level timing tags.
            text = re.sub(r"<[^>]+>", "", raw_text)
            text = (
                text.replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&nbsp;", " ")
                    .replace("&#39;", "'")
                    .replace("&quot;", '"')
                    .strip()
            )
            # Collapse redundant inline whitespace.
            text = re.sub(r"\s+", " ", text).strip()

            if not text or text in seen_texts:
                continue

            seen_texts.add(text)
            raw_entries.append({"start": start_str, "end": end_str, "text": text})

        # Second dedupe pass: filter intermediate YouTube rolling-append states.
        # If entry i is a prefix of entry i+1, entry i is intermediate and is discarded.
        # Also discard empty or single-character noise entries.
        if not raw_entries:
            return []

        entries = []
        for i, entry in enumerate(raw_entries):
            text = entry["text"]
            if len(text) < 2:
                continue
            # Check whether the next few entries start with this text, a rolling-append signal.
            is_intermediate = False
            for j in range(i + 1, min(i + 4, len(raw_entries))):
                next_text = raw_entries[j]["text"]
                if next_text.startswith(text) and len(next_text) > len(text):
                    is_intermediate = True
                    break
            if not is_intermediate:
                entries.append(entry)

        return entries

    def _parse_srt(self, filepath: str) -> list:
        """Parse an SRT subtitle file and return deduplicated entries."""
        entries = []
        seen_texts: set = set()

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read SRT file: {e}")
            return []

        blocks = re.split(r"\n{2,}", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            timing_idx = next((i for i, l in enumerate(lines) if "-->" in l), -1)
            if timing_idx < 0:
                continue

            timing_line = lines[timing_idx]
            text_lines = lines[timing_idx + 1:]

            match = re.match(
                r"(\d{1,2}:\d{2}:\d{2}[.,]\d+)\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d+)",
                timing_line,
            )
            if not match:
                continue

            start_str = self._normalize_time(match.group(1))
            end_str = self._normalize_time(match.group(2))

            text = " ".join(text_lines)
            text = re.sub(r"<[^>]+>", "", text).strip()

            if not text or text in seen_texts:
                continue

            seen_texts.add(text)
            entries.append({"start": start_str, "end": end_str, "text": text})

        return entries

    def _normalize_time(self, time_str: str) -> str:
        """Normalize HH:MM:SS.mmm or MM:SS.mmm to MM:SS."""
        time_str = re.sub(r"[.,]\d+$", "", time_str)
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{h * 60 + m:02d}:{s:02d}"
        elif len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            return f"{m:02d}:{s:02d}"
        return time_str

    def _format_subtitle_entries(self, entries: list, language: str) -> str:
        """Format subtitle entries as Whisper-compatible Markdown for downstream processing."""
        lines = [
            "# Video Transcription",
            "",
            f"**Detected Language:** {language}",
            "**Language Probability:** 1.00",
            "",
            "## Transcription Content",
            "",
        ]
        for entry in entries:
            lines.append(f"**[{entry['start']} - {entry['end']}]**")
            lines.append("")
            lines.append(entry["text"])
            lines.append("")
        return "\n".join(lines)

    async def download_and_convert(
        self,
        url: str,
        output_dir: Path,
        prefetched_title: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Download a video and convert it to m4a.

        prefetched_title: when the caller already probed video info through
        fetch_subtitles, pass the title here to skip a duplicate extract_info
        network request.
        """
        try:
            # Create output directory.
            output_dir.mkdir(exist_ok=True)
            
            # Generate a unique filename.
            unique_id = str(uuid.uuid4())[:8]
            output_template = str(output_dir / f"audio_{unique_id}.%(ext)s")
            
            # Update yt-dlp options.
            ydl_opts = self.ydl_opts.copy()
            ydl_opts['outtmpl'] = output_template
            
            logger.info(f"Starting video download: {url}")
            
            import asyncio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                if prefetched_title:
                    # Title and duration were already fetched by fetch_subtitles; download directly.
                    video_title = prefetched_title
                    expected_duration = 0
                    logger.info(f"Reusing prefetched title and skipping extract_info: {video_title}")
                else:
                    # Fetch video info in a worker thread to avoid blocking the event loop.
                    info = await asyncio.to_thread(ydl.extract_info, url, False)
                    video_title = info.get('title', 'unknown')
                    expected_duration = info.get('duration') or 0
                    logger.info(f"Video title: {video_title}")
                
                # Download in a worker thread to avoid blocking the event loop.
                await asyncio.to_thread(ydl.download, [url])
            
            # Locate the generated m4a file.
            audio_file = str(output_dir / f"audio_{unique_id}.m4a")
            
            if not os.path.exists(audio_file):
                # If the m4a file does not exist, look for other audio formats.
                for ext in ['webm', 'mp4', 'mp3', 'wav']:
                    potential_file = str(output_dir / f"audio_{unique_id}.{ext}")
                    if os.path.exists(potential_file):
                        audio_file = potential_file
                        break
                else:
                    raise Exception("Downloaded audio file was not found")
            
            # Validate duration and try one FFmpeg remux if it differs greatly from the source.
            try:
                import subprocess, shlex
                probe_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(audio_file)}"
                out = subprocess.check_output(probe_cmd, shell=True).decode().strip()
                actual_duration = float(out) if out else 0.0
            except Exception as _:
                actual_duration = 0.0
            
            if expected_duration and actual_duration and abs(actual_duration - expected_duration) / expected_duration > 0.1:
                logger.warning(
                    f"Audio duration looks wrong: expected {expected_duration}s, got {actual_duration}s. Trying a remux repair..."
                )
                try:
                    fixed_path = str(output_dir / f"audio_{unique_id}_fixed.m4a")
                    fix_cmd = f"ffmpeg -y -i {shlex.quote(audio_file)} -vn -c:a aac -b:a 160k -movflags +faststart {shlex.quote(fixed_path)}"
                    subprocess.check_call(fix_cmd, shell=True)
                    # Replace with the repaired file.
                    audio_file = fixed_path
                    # Probe again.
                    out2 = subprocess.check_output(probe_cmd.replace(shlex.quote(audio_file.rsplit('.',1)[0]+'.m4a'), shlex.quote(audio_file)), shell=True).decode().strip()
                    actual_duration2 = float(out2) if out2 else 0.0
                    logger.info(f"Remux complete, new duration is about {actual_duration2:.2f}s")
                except Exception as e:
                    logger.error(f"Remux failed: {e}")
            
            logger.info(f"Audio file saved: {audio_file}")
            return audio_file, video_title
            
        except Exception as e:
            logger.error(f"Video download failed: {str(e)}")
            raise Exception(f"Video download failed: {str(e)}")
    
    def get_video_info(self, url: str) -> dict:
        """
        Get video information.
        
        Args:
            url: video URL
            
        Returns:
            Video information dictionary.
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                }
        except Exception as e:
            logger.error(f"Failed to get video information: {str(e)}")
            raise Exception(f"Failed to get video information: {str(e)}")
