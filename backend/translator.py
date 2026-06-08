import asyncio
import logging
import os
import re
from typing import Optional

from openai import OpenAI

from llm_requests import create_chat_completion
from llm_sanitize import strip_llm_artifacts

logger = logging.getLogger(__name__)


class Translator:
    """Text translator supporting environment or request-provided API key/base URL."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.client = None
        default_model = (
            (os.getenv("OPENAI_TRANSLATION_MODEL") or "").strip()
            or (os.getenv("LOCAL_MODEL_NAME") or "").strip()
            or "gpt-4o"
        )
        self._translation_model = model or default_model

        self.language_map = {
            "zh": "Chinese (Simplified)",
            "zh-tw": "Chinese (Traditional)",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
        }

        eff_key = (api_key.strip() if isinstance(api_key, str) and api_key.strip() else None) or os.getenv(
            "OPENAI_API_KEY"
        )
        if isinstance(api_key, str) and api_key.strip():
            eff_base = (base_url or "").strip().rstrip("/") or os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            )
        else:
            eff_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not eff_key:
            logger.warning("No usable OpenAI API key is set; translation is unavailable")
            return

        try:
            self.client = OpenAI(api_key=eff_key, base_url=eff_base)
            logger.info("Translator OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def _detect_source_language(self, text: str) -> str:
        """Detect the source text language."""
        # Simple metadata-based language detection.
        legacy_detected_language_label = "**\u68c0\u6d4b\u8bed\u8a00:**"
        if "**Detected Language:**" in text or legacy_detected_language_label in text:
            lines = text.split('\n')
            for line in lines:
                if "**Detected Language:**" in line or legacy_detected_language_label in line:
                    lang = line.split(":")[-1].strip()
                    return lang
        
        # Simple character-count based detection.
        total_chars = len(text)
        if total_chars == 0:
            return "en"
        
        # Count Chinese characters.
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        chinese_ratio = chinese_chars / total_chars
        
        # Count Japanese kana.
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        japanese_ratio = japanese_chars / total_chars
        
        # Count Korean Hangul.
        korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
        korean_ratio = korean_chars / total_chars
        
        if chinese_ratio > 0.1:
            return "zh"
        elif japanese_ratio > 0.05:
            return "ja"
        elif korean_ratio > 0.05:
            return "ko"
        else:
            return "en"

    def _normalize_lang_code(self, code: str) -> str:
        if not code:
            return ""
        c = str(code).lower().strip()
        if c.startswith("zh"):
            return "zh"
        if len(c) >= 2 and c[:2] in self.language_map:
            return c[:2]
        return c

    def normalize_lang_code(self, code: Optional[str]) -> str:
        """Public language-code normalizer matching should_translate behavior."""
        return self._normalize_lang_code(code or "")

    def infer_language_code(self, text: str) -> str:
        """Infer an ISO-style language code from text when transcript metadata is missing."""
        return self._detect_source_language(text or "")

    def should_translate(self, source_language: str, target_language: str) -> bool:
        """Return whether translation is needed."""
        if not source_language or not target_language:
            return False

        source_lang = self._normalize_lang_code(source_language)
        target_lang = self._normalize_lang_code(target_language)

        if source_lang == target_lang:
            return False

        chinese_variants = ["zh", "zh-cn", "zh-hans", "chinese"]
        if source_lang in chinese_variants and target_lang in chinese_variants:
            return False

        return True

    def languages_differ_for_translation(self, source_code: Optional[str], summary_lang: Optional[str]) -> bool:
        """Return True when the selected summary language differs from the source language."""
        s = self.normalize_lang_code(source_code or "")
        t = self.normalize_lang_code(summary_lang or "")
        return bool(s and t and self.should_translate(s, t))

    def _smart_chunk_text(self, text: str, max_chars_per_chunk: int = 4000) -> list:
        """Split text into chunks for translation."""
        chunks = []

        # Split by paragraphs first.
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        current_chunk = ""

        for paragraph in paragraphs:
            # Start a new chunk if adding this paragraph would exceed the limit.
            if len(current_chunk) + len(paragraph) + 2 > max_chars_per_chunk and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the final chunk.
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If a chunk is still too long, split it further by sentence.
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars_per_chunk:
                final_chunks.append(chunk)
            else:
                # Split by sentence.
                sentences = re.split(r'[.!?。！？]\s+', chunk)
                current_sub_chunk = ""

                for sentence in sentences:
                    if len(current_sub_chunk) + len(sentence) + 2 > max_chars_per_chunk and current_sub_chunk:
                        final_chunks.append(current_sub_chunk.strip())
                        current_sub_chunk = sentence
                    else:
                        if current_sub_chunk:
                            current_sub_chunk += ". " + sentence
                        else:
                            current_sub_chunk = sentence

                if current_sub_chunk.strip():
                    final_chunks.append(current_sub_chunk.strip())

        return final_chunks

    async def translate_text(self, text: str, target_language: str, source_language: Optional[str] = None) -> str:
        """
        Translate text into the target language.
        
        Args:
            text: text to translate
            target_language: target language code
            source_language: optional source language code; auto-detected if omitted
            
        Returns:
            Translated text.
        """
        try:
            if not self.client:
                logger.warning("OpenAI API is unavailable; cannot translate")
                return text
            
            # Detect source language.
            if not source_language:
                source_language = self._detect_source_language(text)
            
            # Return unchanged text when source and target are the same.
            src_n = self._normalize_lang_code(source_language or "")
            tgt_n = self._normalize_lang_code(target_language)
            if src_n and tgt_n and src_n == tgt_n:
                return text
            
            source_lang_name = self.language_map.get(src_n, self.language_map.get(source_language, source_language))
            target_lang_name = self.language_map.get(tgt_n, self.language_map.get(target_language, target_language))
            
            logger.info(f"Starting translation: {source_lang_name} -> {target_lang_name}")
            
            # Estimate text length and decide whether chunking is needed.
            if len(text) > 3000:
                logger.info(f"Text is long ({len(text)} chars); using chunked translation")
                return await self._translate_with_chunks(text, target_lang_name, source_lang_name)
            else:
                return await self._translate_single_text(text, target_lang_name, source_lang_name)
                
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return text
    
    async def _translate_single_text(self, text: str, target_lang_name: str, source_lang_name: str) -> str:
        """Translate a single text block."""
        system_prompt = f"""You are a professional translator. Translate the {source_lang_name} text accurately into {target_lang_name}.

Translation requirements:
- Preserve the original format and structure, including paragraph breaks and headings.
- Convey the meaning accurately in natural, fluent language.
- Preserve technical terminology accurately.
- Do not add explanations or notes.
- Preserve Markdown formatting when present.
- Output only the translated body: no preface, no closing note, no courtesy text, and no meta-commentary."""

        user_prompt = f"""Translate the following {source_lang_name} text into {target_lang_name}:

{text}

Return only the translation, without any explanation."""

        try:
            response = await create_chat_completion(
                self.client,
                model=self._translation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            return strip_llm_artifacts(response.choices[0].message.content or "")
        except Exception as e:
            logger.error(f"Single-text translation failed: {e}")
            return text
    
    async def _translate_with_chunks(self, text: str, target_lang_name: str, source_lang_name: str) -> str:
        """Translate long text in chunks."""
        chunks = self._smart_chunk_text(text, max_chars_per_chunk=4000)
        logger.info(f"Split text into {len(chunks)} chunks for translation")
        
        async def translate_chunk(i: int, chunk: str) -> str:
            logger.info(f"Translating chunk {i+1}/{len(chunks)}...")

            system_prompt = f"""You are a professional translator. Translate the {source_lang_name} text accurately into {target_lang_name}.

This is part {i+1} of {len(chunks)} of the full document.

Translation requirements:
- Preserve the original format and structure.
- Convey the meaning accurately in natural, fluent language.
- Preserve technical terminology accurately.
- Do not add explanations or notes.
- Keep continuity with neighboring parts.
- Output only the translated body, with no closing note or meta-commentary."""

            user_prompt = f"""Translate the following {source_lang_name} text into {target_lang_name}:

{chunk}

Return only the translation."""

            try:
                response = await create_chat_completion(
                    self.client,
                    model=self._translation_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                translated_chunk = response.choices[0].message.content or ""
                return strip_llm_artifacts(translated_chunk)
            except Exception as e:
                logger.error(f"Translation failed for chunk {i+1}: {e}")
                # Preserve the source chunk on failure.
                return chunk

        translated_chunks = await asyncio.gather(
            *(translate_chunk(i, chunk) for i, chunk in enumerate(chunks))
        )
        
        # Merge translation results.
        return strip_llm_artifacts("\n\n".join(translated_chunks))
