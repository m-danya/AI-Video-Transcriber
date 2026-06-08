import os
import openai
import asyncio
import logging
import re
from typing import Optional

from llm_requests import create_chat_completion
from llm_sanitize import strip_llm_artifacts

logger = logging.getLogger(__name__)


class InvalidSummaryResponse(RuntimeError):
    """Raised when the LLM returns a template/no-source answer instead of a summary."""


class Summarizer:
    """Text summarizer that uses the OpenAI API for multilingual summaries."""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        Initialize the summarizer.

        Priority: explicit arguments > environment variables.
        When model is provided, it is used for both fast_model and advanced_model.
        """
        effective_key = api_key or os.getenv("OPENAI_API_KEY")
        effective_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not effective_key:
            logger.warning("OPENAI_API_KEY is not set; summary features will be unavailable")

        if effective_key:
            kwargs = {"api_key": effective_key}
            if effective_url:
                kwargs["base_url"] = effective_url
                logger.info(f"OpenAI client initialized, base_url={effective_url}")
            else:
                logger.info("OpenAI client initialized with the default endpoint")
            self.client = openai.OpenAI(**kwargs)
        else:
            self.client = None

        default_model = (os.getenv("LOCAL_MODEL_NAME") or "").strip() or None

        # Let the frontend override the environment and hard-coded default model names.
        self.fast_model     = model or default_model or "gpt-3.5-turbo"
        self.advanced_model = model or default_model or "gpt-4o"
        self.summary_retries = self._read_positive_int_env("SUMMARY_LLM_RETRIES", 3)
        
        # Supported language mapping.
        self.language_map = {
            "en": "English",
            "zh": "Chinese (Simplified)",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic"
        }

    def _read_positive_int_env(self, name: str, default: int) -> int:
        raw_value = (os.getenv(name) or "").strip()
        if not raw_value:
            return default
        try:
            return max(1, int(raw_value))
        except ValueError:
            logger.warning("%s=%r is invalid, using %s", name, raw_value, default)
            return default
    
    async def optimize_transcript(self, raw_transcript: str) -> str:
        """
        Optimize transcript text: fix obvious transcription errors and split by meaning.
        Long text is processed in chunks automatically.
        
        Args:
            raw_transcript: raw transcript text
            
        Returns:
            Optimized transcript text in Markdown format.
        """
        try:
            if not self.client:
                logger.warning("OpenAI API is unavailable; returning the raw transcript")
                return raw_transcript

            # Preprocess: remove only timestamps and metadata, preserving spoken/repeated content.
            preprocessed = self._remove_timestamps_and_meta(raw_transcript)
            # JS strategy: chunk by character length to stay close to token limits without estimation drift.
            detected_lang_code = self._detect_transcript_language(preprocessed)
            max_chars_per_chunk = 4000  # Match JS: about 4000 chars per chunk.

            if len(preprocessed) > max_chars_per_chunk:
                logger.info(f"Text is long ({len(preprocessed)} chars); using chunked optimization")
                return await self._format_long_transcript_in_chunks(preprocessed, detected_lang_code, max_chars_per_chunk)
            else:
                return await self._format_single_chunk(preprocessed, detected_lang_code)

        except Exception as e:
            logger.error(f"Transcript optimization failed: {str(e)}")
            logger.info("Returning the raw transcript text")
            return raw_transcript

    def _estimate_tokens(self, text: str) -> int:
        """
        Conservative token-count estimate that includes system prompt and formatting overhead.
        """
        # Conservative estimate that accounts for token expansion in real use.
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_words = len([word for word in text.split() if word.isascii() and word.isalpha()])
        
        # Base token estimate.
        base_tokens = chinese_chars * 1.5 + english_words * 1.3
        
        # Markdown, timestamps, and formatting overhead.
        format_overhead = len(text) * 0.15
        
        # System prompt overhead, roughly 2000-3000 tokens.
        system_prompt_overhead = 2500
        
        total_estimated = int(base_tokens + format_overhead + system_prompt_overhead)
        
        return total_estimated

    async def _optimize_single_chunk(self, raw_transcript: str) -> str:
        """
        Optimize a single text chunk.
        """
        detected_lang = self._detect_transcript_language(raw_transcript)
        lang_instruction = self._get_language_instruction(detected_lang)
        
        system_prompt = f"""You are a professional transcript editor. Optimize the provided video transcript.

Important: this may be an interview, conversation, or speech. If multiple speakers are present, preserve each speaker's original perspective.

Requirements:
1. **Strictly keep the original language ({lang_instruction}); never translate into another language.**
2. **Remove all timestamp markers, such as [00:00 - 00:05].**
3. **Identify and recombine complete sentences split by timestamps.** Merge grammatically incomplete fragments with context.
4. Fix obvious typos and grammar errors.
5. Split complete reconstructed sentences into natural paragraphs by semantic and logical meaning.
6. Separate paragraphs with blank lines.
7. **Preserve the original meaning exactly; do not add or delete real content.**
8. **Never change pronouns or speaker perspective.**
9. Preserve each speaker's original viewpoint and context.
10. Recognize dialogue structure: interviewers may say "you" while interviewees say "I/we"; never confuse them.
11. Ensure each sentence is grammatically complete and natural.

Processing strategy:
- Identify incomplete sentence fragments first, such as fragments ending in a preposition, conjunction, or adjective.
- Inspect neighboring fragments and merge them into complete sentences.
- Re-punctuate so each sentence is grammatically complete.
- Re-paragraph by topic and logic.

Paragraphing requirements:
- Split by topic and logical meaning; each paragraph should contain 1-8 related sentences.
- Keep each paragraph under 400 characters.
- Avoid too many short paragraphs; merge related content.
- Break after a complete thought or viewpoint has been expressed.

Output format:
- Plain text paragraphs, with no timestamps or formatting markers.
- Complete sentence structure.
- One primary topic per paragraph.
- Blank lines between paragraphs.

Critical reminder: this is {lang_instruction} content. Optimize entirely in {lang_instruction}, focusing on incoherence caused by timestamp splitting. Use reasonable paragraphing and avoid overlong paragraphs.

**Critical rule: this may be interview dialogue. Never change any pronoun or speaker perspective. Preserve interviewer "you" and interviewee "I/we" exactly.**"""

        user_prompt = f"""Optimize the following {lang_instruction} video transcript into fluent paragraph text:

{raw_transcript}

Main tasks:
1. Remove all timestamp markers.
2. Identify and recombine complete sentences that were split.
3. Ensure each sentence is grammatically complete and coherent.
4. Re-paragraph by meaning, with blank lines between paragraphs.
5. Keep the language as {lang_instruction}.

Paragraphing guidance:
- Split by topic and logical meaning; each paragraph should contain 1-8 related sentences.
- Keep each paragraph under 400 characters.
- Avoid too many short paragraphs; merge related content.
- Ensure clear blank lines between paragraphs.

Pay special attention to fixing incomplete sentences caused by timestamp splitting and use reasonable paragraph breaks."""

        response = await create_chat_completion(
            self.client,
            model=self.fast_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return strip_llm_artifacts(response.choices[0].message.content or "")

    async def _optimize_with_chunks(self, raw_transcript: str, max_tokens: int) -> str:
        """
        Optimize long text in chunks.
        """
        detected_lang = self._detect_transcript_language(raw_transcript)
        lang_instruction = self._get_language_instruction(detected_lang)
        
        # Split the raw transcript into chunks; timestamps can still guide boundaries.
        chunks = self._split_into_chunks(raw_transcript, max_tokens)
        logger.info(f"Split text into {len(chunks)} chunks for processing")
        
        async def optimize_chunk(i: int, chunk: str) -> str:
            logger.info(f"Optimizing chunk {i+1}/{len(chunks)}...")

            system_prompt = f"""You are a professional text editor. Lightly optimize this transcript chunk.

This is part {i+1} of {len(chunks)} of the full transcript.

Light optimization requirements:
1. **Strictly keep the original language ({lang_instruction}); do not translate.**
2. **Only fix obvious typos and grammar errors.**
3. **Slightly improve sentence flow**, but do not heavily rewrite.
4. **Preserve the original structure and length**; do not perform complex paragraph restructuring.
5. **Preserve the meaning 100%.**

Note: this is only an initial cleanup. Do not do complex rewriting or reorganization."""

            user_prompt = f"""Lightly optimize the following {lang_instruction} text chunk, fixing only typos and grammar:

{chunk}

Output the cleaned text while preserving the original structure."""

            try:
                response = await create_chat_completion(
                    self.client,
                    model=self.fast_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                optimized_chunk = strip_llm_artifacts(response.choices[0].message.content or "")
                return optimized_chunk
                
            except Exception as e:
                logger.error(f"Failed to optimize chunk {i+1}: {e}")
                # Use basic cleanup on failure.
                return self._basic_transcript_cleanup(chunk)

        optimized_chunks = await asyncio.gather(
            *(optimize_chunk(i, chunk) for i, chunk in enumerate(chunks))
        )
        
        # Merge all optimized chunks.
        merged_text = "\n\n".join(optimized_chunks)
        
        # Run a second paragraph organization pass on the merged text.
        logger.info("Running final paragraph organization...")
        final_result = await self._final_paragraph_organization(merged_text, lang_instruction)
        
        logger.info("Chunked optimization complete")
        return final_result

    # ===== Port from JS openaiService.js: chunking/context/dedupe/formatting =====

    def _ensure_markdown_paragraphs(self, text: str) -> str:
        """Ensure Markdown paragraph spacing, heading spacing, and compact blank lines."""
        if not text:
            return text
        formatted = text.replace("\r\n", "\n")
        import re
        # Add a blank line after headings.
        formatted = re.sub(r"(^#{1,6}\s+.*)\n([^\n#])", r"\1\n\n\2", formatted, flags=re.M)
        # Collapse 3+ line breaks to 2.
        formatted = re.sub(r"\n{3,}", "\n\n", formatted)
        # Trim leading/trailing blank lines.
        formatted = re.sub(r"^\n+", "", formatted)
        formatted = re.sub(r"\n+$", "", formatted)
        return formatted

    async def _format_single_chunk(self, chunk_text: str, transcript_language: str = 'zh') -> str:
        """Optimize one chunk by correcting and formatting within the 4000-token limit."""
        # Build system/user prompts matching the JS version.
        if transcript_language == 'zh':
            prompt = (
                "Intelligently optimize and format the following audio transcript text:\n\n"
                "**Content optimization (accuracy first):**\n"
                "1. Fix transcription errors, typos, homophones, and proper nouns.\n"
                "2. Moderately improve grammar and complete incomplete sentences while preserving meaning and language.\n"
                "3. Spoken-language handling: keep natural fillers and repetitions; do not remove content; add only necessary punctuation.\n"
                "4. **Never change pronouns or speaker perspective.**\n\n"
                "**Paragraphing rules:**\n"
                "- Split by topic and logical meaning; each paragraph should contain 1-8 related sentences.\n"
                "- Keep each paragraph under 400 characters.\n"
                "- Avoid too many short paragraphs; merge related content.\n\n"
                "**Format:** Markdown paragraphs with blank lines between paragraphs.\n\n"
                f"Original transcript text:\n{chunk_text}"
            )
            system_prompt = (
                "You are a professional transcript optimization assistant. Fix errors, improve fluency, and format the transcript. "
                "You must preserve meaning and must not remove spoken language, repetitions, or details; only timestamps or metadata may be removed. "
                "Never change pronouns or speaker perspective. This may be interview dialogue: the interviewer uses 'you' and the interviewee uses 'I/we'."
            )
        else:
            prompt = (
                "Please intelligently optimize and format the following audio transcript text:\n\n"
                "Content Optimization (Accuracy First):\n"
                "1. Error Correction (typos, homophones, proper nouns)\n"
                "2. Moderate grammar improvement, complete incomplete sentences, keep original language/meaning\n"
                "3. Speech processing: keep natural fillers and repetitions, do NOT remove content; only add punctuation if needed\n"
                "4. **NEVER change pronouns (I, you, he, she, etc.) or speaker perspective**\n\n"
                "Segmentation Rules: Group 1-8 related sentences per paragraph by topic/logic; paragraph length NOT exceed 400 characters; avoid too many short paragraphs\n\n"
                "Format: Markdown paragraphs with blank lines between paragraphs\n\n"
                f"Original transcript text:\n{chunk_text}"
            )
            system_prompt = (
                "You are a professional transcript formatting assistant. Fix errors and improve fluency "
                "without changing meaning or removing any content; only timestamps/meta may be removed; keep Markdown paragraphs with blank lines. "
                "NEVER change pronouns or speaker perspective. This may be an interview: interviewer uses 'you', interviewee uses 'I/we'."
            )

        try:
            response = await create_chat_completion(
                self.client,
                model=self.fast_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            optimized_text = strip_llm_artifacts(response.choices[0].message.content or "")
            # Remove headings such as "# Transcript" / "## Transcript".
            optimized_text = self._remove_transcript_heading(optimized_text)
            enforced = self._enforce_paragraph_max_chars(optimized_text.strip(), max_chars=400)
            return self._ensure_markdown_paragraphs(enforced)
        except Exception as e:
            logger.error(f"Single-chunk text optimization failed: {e}")
            return self._apply_basic_formatting(chunk_text)

    def _smart_split_long_chunk(self, text: str, max_chars_per_chunk: int) -> list:
        """Split oversized text safely at sentence or space boundaries."""
        chunks = []
        pos = 0
        while pos < len(text):
            end = min(pos + max_chars_per_chunk, len(text))
            if end < len(text):
                # Prefer sentence boundaries.
                sentence_endings = ['。', '！', '？', '.', '!', '?']
                best = -1
                for ch in sentence_endings:
                    idx = text.rfind(ch, pos, end)
                    if idx > best:
                        best = idx
                if best > pos + int(max_chars_per_chunk * 0.7):
                    end = best + 1
                else:
                    # Fallback to space boundaries.
                    space_idx = text.rfind(' ', pos, end)
                    if space_idx > pos + int(max_chars_per_chunk * 0.8):
                        end = space_idx
            chunks.append(text[pos:end].strip())
            pos = end
        return [c for c in chunks if c]

    def _find_safe_cut_point(self, text: str) -> int:
        """Find a safe cut point: paragraph, then sentence, then phrase."""
        import re
        # Paragraph.
        p = text.rfind("\n\n")
        if p > 0:
            return p + 2
        # Sentence.
        last_sentence_end = -1
        for m in re.finditer(r"[。！？\.!?]\s*", text):
            last_sentence_end = m.end()
        if last_sentence_end > 20:
            return last_sentence_end
        # Phrase.
        last_phrase_end = -1
        for m in re.finditer(r"[，；,;]\s*", text):
            last_phrase_end = m.end()
        if last_phrase_end > 20:
            return last_phrase_end
        return len(text)

    def _find_overlap_between_texts(self, text1: str, text2: str) -> str:
        """Detect overlap between adjacent chunks for deduplication."""
        max_len = min(len(text1), len(text2))
        # Try from longest to shortest.
        for length in range(max_len, 19, -1):
            suffix = text1[-length:]
            prefix = text2[:length]
            if suffix == prefix:
                cut = self._find_safe_cut_point(prefix)
                if cut > 20:
                    return prefix[:cut]
                return suffix
        return ""

    def _apply_basic_formatting(self, text: str) -> str:
        """Fallback when AI fails: combine sentences into paragraphs separated by blank lines."""
        if not text or not text.strip():
            return text
        import re
        parts = re.split(r"([。！？\.!?]+\s*)", text)
        sentences = []
        current = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                current += part
            else:
                current += part
                if current.strip():
                    sentences.append(current.strip())
                    current = ""
        if current.strip():
            sentences.append(current.strip())
        paras = []
        cur = ""
        sentence_count = 0
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            sentence_count += 1
            # Improved paragraphing logic: consider sentence count and length.
            should_break = False
            if len(candidate) > 400 and cur:  # Paragraph too long.
                should_break = True
            elif len(candidate) > 200 and sentence_count >= 3:  # Medium length with enough sentences.
                should_break = True
            elif sentence_count >= 6:  # Too many sentences.
                should_break = True
            
            if should_break:
                paras.append(cur.strip())
                cur = s
                sentence_count = 1
            else:
                cur = candidate
        if cur.strip():
            paras.append(cur.strip())
        return self._ensure_markdown_paragraphs("\n\n".join(paras))

    async def _format_long_transcript_in_chunks(self, raw_transcript: str, transcript_language: str, max_chars_per_chunk: int) -> str:
        """Build optimized text using smart chunking, context, and dedupe from the JS strategy."""
        import re
        # Split by sentence first and build chunks under max_chars_per_chunk.
        parts = re.split(r"([。！？\.!?]+\s*)", raw_transcript)
        sentences = []
        buf = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                buf += part
            else:
                buf += part
                if buf.strip():
                    sentences.append(buf.strip())
                    buf = ""
        if buf.strip():
            sentences.append(buf.strip())

        chunks = []
        cur = ""
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            if len(candidate) > max_chars_per_chunk and cur:
                chunks.append(cur.strip())
                cur = s
            else:
                cur = candidate
        if cur.strip():
            chunks.append(cur.strip())

        # Safely split chunks that are still too long.
        final_chunks = []
        for c in chunks:
            if len(c) <= max_chars_per_chunk:
                final_chunks.append(c)
            else:
                final_chunks.extend(self._smart_split_long_chunk(c, max_chars_per_chunk))

        logger.info(f"Text split into {len(final_chunks)} chunks for processing")

        async def format_chunk(i: int, c: str) -> str:
            chunk_with_context = c
            if i > 0:
                prev_tail = final_chunks[i - 1][-100:]
                marker = f"[Context continued: {prev_tail}]"
                chunk_with_context = marker + "\n\n" + c
            try:
                oc = await self._format_single_chunk(chunk_with_context, transcript_language)
                # Remove the context marker.
                oc = re.sub(r"^\[Context continued:.*?\]\s*", "", oc, flags=re.S)
                return oc
            except Exception as e:
                logger.warning(f"Chunk {i+1} optimization failed; using basic formatting: {e}")
                return self._apply_basic_formatting(c)

        optimized = await asyncio.gather(
            *(format_chunk(i, c) for i, c in enumerate(final_chunks))
        )

        # Deduplicate adjacent chunks.
        deduped = []
        for i, c in enumerate(optimized):
            cur_txt = c
            if i > 0 and deduped:
                prev = deduped[-1]
                overlap = self._find_overlap_between_texts(prev[-200:], cur_txt[:200])
                if overlap:
                    cur_txt = cur_txt[len(overlap):].lstrip()
                    if not cur_txt:
                        continue
            if cur_txt.strip():
                deduped.append(cur_txt)

        merged = "\n\n".join(deduped)
        merged = self._remove_transcript_heading(merged)
        enforced = self._enforce_paragraph_max_chars(merged, max_chars=400)
        return self._ensure_markdown_paragraphs(enforced)

    def _remove_timestamps_and_meta(self, text: str) -> str:
        """Remove only timestamp lines and obvious metadata, preserving spoken/repeated content."""
        legacy_detected_language_label = "**\u68c0\u6d4b\u8bed\u8a00:**"
        legacy_language_probability_label = "**\u8bed\u8a00\u6982\u7387:**"
        lines = text.split('\n')
        kept = []
        for line in lines:
            s = line.strip()
            # Skip timestamps and metadata.
            if (s.startswith('**[') and s.endswith(']**')):
                continue
            if s.startswith('# '):
                # Skip the top-level title; it is usually the video title and is added back later.
                continue
            if (
                s.startswith('**Detected Language:**')
                or s.startswith('**Language Probability:**')
                or s.startswith(legacy_detected_language_label)
                or s.startswith(legacy_language_probability_label)
            ):
                continue
            kept.append(line)
        # Normalize blank lines.
        cleaned = '\n'.join(kept)
        return cleaned

    def _enforce_paragraph_max_chars(self, text: str, max_chars: int = 400) -> str:
        """Split paragraphs and ensure each paragraph stays under max_chars."""
        if not text:
            return text
        import re
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p is not None]
        new_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) <= max_chars:
                new_paragraphs.append(para)
                continue
            # Sentence split.
            parts = re.split(r"([。！？\.!?]+\s*)", para)
            sentences = []
            buf = ""
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    buf += part
                else:
                    buf += part
                    if buf.strip():
                        sentences.append(buf.strip())
                        buf = ""
            if buf.strip():
                sentences.append(buf.strip())
            cur = ""
            for s in sentences:
                candidate = (cur + (" " if cur else "") + s).strip()
                if len(candidate) > max_chars and cur:
                    new_paragraphs.append(cur)
                    cur = s
                else:
                    cur = candidate
            if cur:
                new_paragraphs.append(cur)
        return "\n\n".join([p.strip() for p in new_paragraphs if p is not None])

    def _remove_transcript_heading(self, text: str) -> str:
        """Remove lines headed as Transcript without changing body text."""
        if not text:
            return text
        import re
        # Remove headings such as '## Transcript', '# Transcript Text', or '### transcript'.
        lines = text.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^#{1,6}\s*transcript(\s+text)?\s*$", stripped, flags=re.I):
                continue
            filtered.append(line)
        return '\n'.join(filtered)

    def _split_into_chunks(self, text: str, max_tokens: int) -> list:
        """
        Split raw transcript text into suitably sized chunks.
        Strategy: extract plain text first, then split naturally by sentence and paragraph.
        """
        import re
        
        # 1. Extract plain text first, removing timestamps and headings.
        pure_text = self._extract_pure_text(text)
        
        # 2. Split by sentence while preserving sentence integrity.
        sentences = self._split_into_sentences(pure_text)
        
        # 3. Assemble chunks under the token limit.
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # Check whether the sentence can fit in the current chunk.
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Current chunk is full; save it and start a new one.
                chunks.append(self._join_sentences(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                # Add to the current chunk.
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add the final chunk.
        if current_chunk:
            chunks.append(self._join_sentences(current_chunk))
        
        return chunks
    
    def _extract_pure_text(self, raw_transcript: str) -> str:
        """
        Extract plain text from a raw transcript, removing timestamps and metadata.
        """
        lines = raw_transcript.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip timestamps, headings, and metadata.
            legacy_detected_language_label = "**\u68c0\u6d4b\u8bed\u8a00:**"
            legacy_language_probability_label = "**\u8bed\u8a00\u6982\u7387:**"
            if (line.startswith('**[') and line.endswith(']**') or
                line.startswith('#') or
                line.startswith('**Detected Language:**') or
                line.startswith('**Language Probability:**') or
                line.startswith(legacy_detected_language_label) or
                line.startswith(legacy_language_probability_label) or
                not line):
                continue
            text_lines.append(line)
        
        return ' '.join(text_lines)
    
    def _split_into_sentences(self, text: str) -> list:
        """
        Split text into sentences, accounting for English and CJK punctuation.
        """
        import re
        
        # English and CJK sentence endings.
        sentence_endings = r'[.!?。！？;；]+'
        
        # Split sentences while preserving punctuation.
        parts = re.split(f'({sentence_endings})', text)
        
        sentences = []
        current = ""
        
        for i, part in enumerate(parts):
            if re.match(sentence_endings, part):
                # Sentence ending; append to the current sentence.
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                # Sentence content.
                current += part
        
        # Handle the final part when it has no ending punctuation.
        if current.strip():
            sentences.append(current.strip())
        
        return [s for s in sentences if s.strip()]
    

    
    def _join_sentences(self, sentences: list) -> str:
        """
        Recombine sentences into a paragraph.
        """
        return ' '.join(sentences)

    def _basic_transcript_cleanup(self, raw_transcript: str) -> str:
        """
        Basic transcript cleanup: remove timestamps and heading metadata.
        Fallback when GPT optimization fails.
        """
        lines = raw_transcript.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip timestamp lines.
            if line.strip().startswith('**[') and line.strip().endswith(']**'):
                continue
            # Skip heading lines.
            if line.strip().startswith('# ') or line.strip().startswith('## '):
                continue
            # Skip language metadata lines.
            legacy_detected_language_label = "**\u68c0\u6d4b\u8bed\u8a00:**"
            legacy_language_probability_label = "**\u8bed\u8a00\u6982\u7387:**"
            if (
                line.strip().startswith('**Detected Language:**')
                or line.strip().startswith('**Language Probability:**')
                or line.strip().startswith(legacy_detected_language_label)
                or line.strip().startswith(legacy_language_probability_label)
            ):
                continue
            # Keep non-empty text lines.
            if line.strip():
                cleaned_lines.append(line.strip())
        
        # Recombine sentences and paragraph intelligently.
        text = ' '.join(cleaned_lines)
        
        # Smarter sentence handling that accounts for English and CJK punctuation.
        import re
        
        # Split by periods, question marks, and exclamation marks.
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        paragraphs = []
        current_paragraph = []
        
        for i, sentence in enumerate(sentences):
            if sentence:
                current_paragraph.append(sentence)
                
                # Smart paragraphing conditions:
                # 1. A paragraph every 3 sentences as the basic rule.
                # 2. Force breaks at topic-transition keywords.
                # 3. Avoid overlong paragraphs.
                topic_change_keywords = [
                    '\u9996\u5148', '\u5176\u6b21', '\u7136\u540e', '\u63a5\u4e0b\u6765',
                    '\u53e6\u5916', '\u6b64\u5916', '\u6700\u540e', '\u603b\u4e4b',
                    'first', 'second', 'third', 'next', 'also', 'however', 'finally',
                    '\u73b0\u5728', '\u90a3\u4e48', '\u6240\u4ee5', '\u56e0\u6b64',
                    '\u4f46\u662f', '\u7136\u800c',
                    'now', 'so', 'therefore', 'but', 'however'
                ]
                
                should_break = False
                
                # Check whether a paragraph break is needed.
                if len(current_paragraph) >= 3:  # Basic length condition.
                    should_break = True
                elif len(current_paragraph) >= 2:  # Shorter paragraph but topic transition.
                    for keyword in topic_change_keywords:
                        if sentence.lower().startswith(keyword.lower()):
                            should_break = True
                            break
                
                if should_break or len(current_paragraph) >= 4:  # Maximum length limit.
                    # Combine the current paragraph.
                    paragraph_text = '. '.join(current_paragraph)
                    if not paragraph_text.endswith('.'):
                        paragraph_text += '.'
                    paragraphs.append(paragraph_text)
                    current_paragraph = []
        
        # Add remaining sentences.
        if current_paragraph:
            paragraph_text = '. '.join(current_paragraph)
            if not paragraph_text.endswith('.'):
                paragraph_text += '.'
            paragraphs.append(paragraph_text)
        
        return '\n\n'.join(paragraphs)

    async def _final_paragraph_organization(self, text: str, lang_instruction: str) -> str:
        """
        Run final paragraph organization on merged text with prompt guidance and validation.
        """
        try:
            # Estimate text length and chunk if it is too long.
            estimated_tokens = self._estimate_tokens(text)
            if estimated_tokens > 3000:  # Chunk very long text.
                return await self._organize_long_text_paragraphs(text, lang_instruction)
            
            system_prompt = f"""You are a professional {lang_instruction} paragraph organization expert. Reorganize paragraphs by semantics and logic.

Core principles:
1. **Strictly keep the original language ({lang_instruction}); never translate.**
2. **Keep all content complete; do not delete or add information.**
3. **Paragraph by semantic logic:** each paragraph should center on one complete idea or topic.
4. **Control paragraph length:** no paragraph should exceed 250 words.
5. **Keep natural flow:** paragraphs should have logical connections.

Paragraphing criteria:
- **Semantic completeness:** each paragraph covers one complete concept or event.
- **Moderate length:** 3-7 sentences; never exceed 250 words per paragraph.
- **Logical boundaries:** break at topic, time, or viewpoint transitions.
- **Natural breaks:** follow natural pauses and logic from the speaker.

Forbidden:
- Creating huge paragraphs longer than 250 words.
- Forcing unrelated content together.
- Breaking complete stories or arguments.

Output format: separate paragraphs with blank lines."""

            user_prompt = f"""Reorganize the paragraph structure of the following {lang_instruction} text. Segment strictly by semantics and logic, ensuring each paragraph stays under 200 words:

{text}

Re-paragraphed text:"""

            response = await create_chat_completion(
                self.client,
                model=self.advanced_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            organized_text = strip_llm_artifacts(response.choices[0].message.content or "")
            
            # Engineering validation: check paragraph length.
            validated_text = self._validate_paragraph_lengths(organized_text)
            
            return validated_text
            
        except Exception as e:
            logger.error(f"Final paragraph organization failed: {e}")
            # Use basic paragraph handling on failure.
            return self._basic_paragraph_fallback(text)

    async def _organize_long_text_paragraphs(self, text: str, lang_instruction: str) -> str:
        """
        Organize very long text in paragraph chunks.
        """
        try:
            # Split by existing paragraphs.
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            chunk_texts = []
            current_chunk = []
            current_tokens = 0
            max_chunk_tokens = 2500  # Chunk size that fits a 4000-token limit.
            
            for para in paragraphs:
                para_tokens = self._estimate_tokens(para)
                
                if current_tokens + para_tokens > max_chunk_tokens and current_chunk:
                    chunk_texts.append('\n\n'.join(current_chunk))
                    
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens
            
            # Handle the final chunk.
            if current_chunk:
                chunk_texts.append('\n\n'.join(current_chunk))

            organized_chunks = await asyncio.gather(
                *(self._organize_single_chunk(chunk_text, lang_instruction) for chunk_text in chunk_texts)
            )
            
            return '\n\n'.join(organized_chunks)
            
        except Exception as e:
            logger.error(f"Long-text paragraph organization failed: {e}")
            return self._basic_paragraph_fallback(text)

    async def _organize_single_chunk(self, text: str, lang_instruction: str) -> str:
        """
        Organize paragraphs in a single text chunk.
        """
        system_prompt = f"""You are a {lang_instruction} paragraph organization expert. Reorganize paragraphs by semantics, ensuring each paragraph does not exceed 200 words.

Core requirements:
1. Strictly maintain the original {lang_instruction} language
2. Organize by semantic logic, one theme per paragraph
3. Each paragraph must not exceed 250 words
4. Separate paragraphs with blank lines
5. Keep content complete, do not reduce information"""

        user_prompt = f"""Re-paragraph the following text in {lang_instruction}, ensuring each paragraph does not exceed 200 words:

{text}"""

        response = await create_chat_completion(
            self.client,
            model=self.advanced_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return strip_llm_artifacts(response.choices[0].message.content or "")

    def _validate_paragraph_lengths(self, text: str) -> str:
        """
        Validate paragraph length and split overlong paragraphs.
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        validated_paragraphs = []
        
        for para in paragraphs:
            word_count = len(para.split())
            
            if word_count > 300:  # Paragraph exceeds 300 words.
                logger.warning(f"Detected an overlong paragraph ({word_count} words); trying to split it")
                # Try splitting the long paragraph by sentence.
                split_paras = self._split_long_paragraph(para)
                validated_paragraphs.extend(split_paras)
            else:
                validated_paragraphs.append(para)
        
        return '\n\n'.join(validated_paragraphs)

    def _split_long_paragraph(self, paragraph: str) -> list:
        """
        Split an overlong paragraph.
        """
        import re
        
        # Split by sentence.
        sentences = re.split(r'[.!?。！？]\s+', paragraph)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        split_paragraphs = []
        current_para = []
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_words + sentence_words > 200 and current_para:
                # Current paragraph reached the length limit.
                split_paragraphs.append(' '.join(current_para))
                current_para = [sentence]
                current_words = sentence_words
            else:
                current_para.append(sentence)
                current_words += sentence_words
        
        # Add the final paragraph.
        if current_para:
            split_paragraphs.append(' '.join(current_para))
        
        return split_paragraphs

    def _basic_paragraph_fallback(self, text: str) -> str:
        """
        Basic paragraph fallback used when GPT organization fails.
        """
        import re
        
        # Remove redundant blank lines.
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        basic_paragraphs = []
        
        for para in paragraphs:
            word_count = len(para.split())
            
            if word_count > 250:
                # Split long paragraphs by sentence.
                split_paras = self._split_long_paragraph(para)
                basic_paragraphs.extend(split_paras)
            elif word_count < 30 and basic_paragraphs:
                # Merge short paragraphs with the previous paragraph if the result stays under 200 words.
                last_para = basic_paragraphs[-1]
                combined_words = len(last_para.split()) + word_count
                
                if combined_words <= 200:
                    basic_paragraphs[-1] = last_para + ' ' + para
                else:
                    basic_paragraphs.append(para)
            else:
                basic_paragraphs.append(para)
        
        return '\n\n'.join(basic_paragraphs)

    async def summarize(self, transcript: str, target_language: str = "zh", video_title: str = None) -> str:
        """
        Generate a summary for a video transcript.
        
        Args:
            transcript: transcript text
            target_language: target language code
            
        Returns:
            Summary text in Markdown format.
        """
        try:
            if not self.client:
                logger.warning("OpenAI API is unavailable; generating fallback summary")
                return self._generate_fallback_summary(transcript, target_language, video_title)
            
            # Estimate transcript length and decide whether chunked summarization is needed.
            estimated_tokens = self._estimate_tokens(transcript)
            single_pass_max_chars = 12000
            max_summarize_tokens = 4000  # Used only for logging/compatibility in long-text chunking.
            
            if len(transcript) <= single_pass_max_chars:
                logger.info(
                    "Text fits single-pass summarization (%s chars, estimated %s tokens)",
                    len(transcript),
                    estimated_tokens,
                )
                return await self._summarize_single_text(transcript, target_language, video_title)
            else:
                # Long-text chunked summarization.
                logger.info(
                    "Text is long (%s chars, estimated %s tokens); using chunked summarization",
                    len(transcript),
                    estimated_tokens,
                )
                return await self._summarize_with_chunks(transcript, target_language, video_title, max_summarize_tokens)
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return self._generate_fallback_summary(transcript, target_language, video_title)

    async def _summarize_single_text(self, transcript: str, target_language: str, video_title: str = None) -> str:
        """
        Summarize a single text.
        """
        # Get target language name.
        language_name = self.language_map.get(target_language, "Chinese (Simplified)")
        
        # English prompts work for all target languages.
        system_prompt = f"""You are an expert editor. Write a concise EXECUTIVE SUMMARY in {language_name} of the following material.

Hard rules:
- Length: about 180–450 words in {language_name} (use the lower end if the source is short). Never reproduce long verbatim quotes or extended sentence-by-sentence rewrites of the transcript.
- Content: main thesis, 3–7 key takeaways, important conclusions, and critical facts or numbers only. Tight prose; short bullet lists are OK for takeaways.
- Do NOT restate the full transcript, do NOT add preamble ("Here is..."), and do NOT add closings such as offers to revise or "let me know if...".
- Markdown: optional `## Key takeaways` then paragraphs; avoid decorative filler headings.

Output ONLY the summary body in {language_name}."""

        user_prompt = f"""Summarize the following content in {language_name}. Follow the system rules strictly (brief executive summary, no meta-commentary):

{transcript}"""

        logger.info(f"Generating {language_name} summary...")
        
        summary = await self._generate_summary_with_retries(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2200,
            operation_label="single-pass summary",
        )
        if self._looks_like_invalid_summary(summary):
            logger.warning("Single-pass summary remained invalid after retries; using local extractive fallback")
            summary = self._generate_extractive_chunk_summary(
                transcript,
                language_name,
                1,
                1,
            )
            if self._looks_like_invalid_summary(summary):
                return self._generate_fallback_summary(transcript, target_language, video_title)

        return self._format_summary_with_meta(summary, target_language, video_title)

    async def _summarize_with_chunks(self, transcript: str, target_language: str, video_title: str, max_tokens: int) -> str:
        """
        Summarize long text in chunks.
        """
        language_name = self.language_map.get(target_language, "Chinese (Simplified)")

        # JS strategy: smart character-based chunking by paragraphs, then sentences.
        chunks = self._smart_chunk_text(transcript, max_chars_per_chunk=4000)
        logger.info(f"Split text into {len(chunks)} chunks for summarization")
        
        async def summarize_chunk(i: int, chunk: str) -> str:
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}...")

            system_prompt = f"""You are a summarization expert. Write a brief section summary in {language_name}.

This is part {i+1} of {len(chunks)} of the full transcript.

Rules:
- About 80–160 words in {language_name}; bullets OK for key points.
- Do not echo the transcript verbatim; capture only new information in this segment.
- No preamble or meta-closings."""

            user_prompt = f"""[Part {i+1}/{len(chunks)}] Summarize in {language_name} (80–160 words, tight prose):

{chunk}

Output content only, no headings like "Summary:"."""

            chunk_summary = await self._generate_summary_with_retries(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=600,
                operation_label=f"chunk {i+1}/{len(chunks)} summary",
            )
            if self._looks_like_invalid_summary(chunk_summary):
                logger.warning(
                    "Summary chunk %s/%s remained invalid after retries; using local extractive fallback",
                    i + 1,
                    len(chunks),
                )
                chunk_summary = self._generate_extractive_chunk_summary(
                    chunk,
                    language_name,
                    i + 1,
                    len(chunks),
                )
            return chunk_summary

        # Generate a partial summary for each chunk.
        chunk_summaries = await asyncio.gather(
            *(summarize_chunk(i, chunk) for i, chunk in enumerate(chunks))
        )
        
        # Combine all partial summaries with numbering; many chunks are integrated hierarchically.
        combined_summaries = "\n\n".join([f"[Part {idx+1}]\n" + s for idx, s in enumerate(chunk_summaries)])

        logger.info("Integrating final summary...")
        if len(chunk_summaries) > 10:
            final_summary = await self._integrate_hierarchical_summaries(chunk_summaries, target_language)
        else:
            final_summary = await self._integrate_chunk_summaries(combined_summaries, target_language)

        return self._format_summary_with_meta(final_summary, target_language, video_title)

    async def _generate_summary_with_retries(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        operation_label: str,
    ) -> str:
        """Call the LLM for summary text and retry invalid/template responses."""
        last_error = None
        for attempt in range(1, self.summary_retries + 1):
            retry_note = ""
            if attempt > 1:
                retry_note = (
                    "\n\nRetry correction: your previous response was rejected because it looked "
                    "empty, templated, or claimed that source content was not available. The source "
                    "text is present in the user message. Write a real summary of that text only."
                )

            try:
                response = await create_chat_completion(
                    self.client,
                    model=self.advanced_model,
                    messages=[
                        {"role": "system", "content": system_prompt + retry_note},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                summary = strip_llm_artifacts(response.choices[0].message.content or "")
                if self._looks_like_invalid_summary(summary):
                    raise InvalidSummaryResponse(
                        f"LLM returned an invalid/template response for {operation_label}"
                    )
                return summary
            except Exception as e:
                last_error = e
                if attempt < self.summary_retries:
                    logger.warning(
                        "%s attempt %s/%s failed: %s; retrying",
                        operation_label,
                        attempt,
                        self.summary_retries,
                        e,
                    )
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
                else:
                    logger.error(
                        "%s failed after %s attempts: %s",
                        operation_label,
                        self.summary_retries,
                        e,
                    )

        if last_error:
            logger.debug("%s last error: %r", operation_label, last_error)
        return ""

    def _generate_extractive_chunk_summary(
        self,
        chunk: str,
        language_name: str,
        part_number: int,
        total_parts: int,
    ) -> str:
        """Local fallback that keeps the pipeline alive without inventing content."""
        cleaned = self._extract_pure_text(chunk)
        sentences = self._split_into_sentences(cleaned)
        if not sentences:
            cleaned = re.sub(r"\s+", " ", chunk or "").strip()
            if not cleaned:
                return f"[Part {part_number}/{total_parts}]"
            return cleaned[:700].strip()

        selected = sentences[:5]
        summary = " ".join(selected)
        if len(summary) > 900:
            summary = summary[:900].rsplit(" ", 1)[0].strip()
        logger.info(
            "Generated extractive fallback for chunk %s/%s in %s",
            part_number,
            total_parts,
            language_name,
        )
        return summary

    def _smart_chunk_text(self, text: str, max_chars_per_chunk: int = 3500) -> list:
        """Smart chunking by character limit, paragraphs first and then sentences."""
        chunks = []
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        cur = ""
        for p in paragraphs:
            candidate = (cur + "\n\n" + p).strip() if cur else p
            if len(candidate) > max_chars_per_chunk and cur:
                chunks.append(cur.strip())
                cur = p
            else:
                cur = candidate
        if cur.strip():
            chunks.append(cur.strip())

        # Split overlong chunks by sentence in a second pass.
        import re
        final_chunks = []
        for c in chunks:
            if len(c) <= max_chars_per_chunk:
                final_chunks.append(c)
            else:
                sentences = [s.strip() for s in re.split(r"[。！？\.!?]+", c) if s.strip()]
                scur = ""
                for s in sentences:
                    candidate = (scur + '. ' + s).strip() if scur else s
                    if len(candidate) > max_chars_per_chunk and scur:
                        final_chunks.append(scur.strip())
                        scur = s
                    else:
                        scur = candidate
                if scur.strip():
                    final_chunks.append(scur.strip())
        return final_chunks

    async def _integrate_hierarchical_summaries(
        self, chunk_summaries: list, target_language: str
    ) -> str:
        """Many partial summaries: fold through the same integrator as the <=10 case."""
        combined = "\n\n".join(
            f"[Part {idx + 1}]\n{s}" for idx, s in enumerate(chunk_summaries)
        )
        return await self._integrate_chunk_summaries(combined, target_language)

    async def _integrate_chunk_summaries(self, combined_summaries: str, target_language: str) -> str:
        """
        Integrate chunk summaries into one coherent final summary.
        """
        language_name = self.language_map.get(target_language, "Chinese (Simplified)")
        
        try:
            system_prompt = f"""You integrate partial summaries into ONE concise executive summary in {language_name}.

Rules:
- Total length about 280–650 words in {language_name}; remove duplication, do not expand into a transcript-length rewrite.
- Markdown: paragraphs separated by blank lines; optional `## Key takeaways` only if it adds clarity.
- No preamble, no meta-closings (e.g. offers to revise or "let me know")."""

            user_prompt = f"""Merge the following partial summaries into one executive summary in {language_name}:

{combined_summaries}"""

            integrated = await self._generate_summary_with_retries(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=2200,
                operation_label="summary integration",
            )
            if self._looks_like_invalid_summary(integrated):
                logger.warning("Integrated summary looks invalid; returning the valid merged chunk summaries")
                return combined_summaries
            return integrated
        except Exception as e:
            logger.error(f"Summary integration failed: {e}")
            # Fall back to direct merge.
            return combined_summaries

    def _looks_like_invalid_summary(self, text: str) -> bool:
        """Detect common no-input/template answers produced by some LLM backends."""
        if not text or not text.strip():
            return True

        normalized = re.sub(r"\s+", " ", text.strip().lower())
        no_source_patterns = [
            r"\u0444\u0430\u043a\u0442\u0438\u0447\u0435\u0441\u043a\w* "
            r"\u0441\u043e\u0434\u0435\u0440\u0436\u0430\u043d\w*.{0,80}"
            r"\u043d\u0435 \u0431\u044b\u043b\w* "
            r"\u043f\u0440\u0435\u0434\u043e\u0441\u0442\u0430\u0432",
            r"\u0447\u0430\u0441\u0442[\u044c\u0438]\s+\d+(?:\s*,\s*\d+)*"
            r"(?:\s+\u0438\s+\d+)?.{0,80}\u043d\u0435 \u0431\u044b\u043b\w* "
            r"\u043f\u0440\u0435\u0434\u043e\u0441\u0442\u0430\u0432",
            r"content.{0,80}(?:not|wasn['’]?t|isn['’]?t).{0,40}(?:provided|available)",
            r"(?:no|without).{0,30}(?:source|actual)?\s*(?:content|material|transcript)",
        ]
        if any(re.search(pattern, normalized, flags=re.I) for pattern in no_source_patterns):
            return True

        placeholder_patterns = [
            r"\[(?:\u0443\u043a\u0430\u0437\u0430\u0442\u044c|"
            r"\u043d\u0430\u0437\u0432\u0430\u043d\u0438\u0435|"
            r"\u0434\u0430\u0442\u0430|"
            r"\u043e\u0441\u043d\u043e\u0432\u043d|"
            r"\u0432\u0441\u0442\u0430\u0432\u0438\u0442\u044c)[^\]]{0,80}\]",
            r"\[(?:specify|insert|project|name|date|main|segment)[^\]]{0,80}\]",
            r"\b[\u0445x]\s*%",
        ]
        placeholder_hits = sum(
            1 for pattern in placeholder_patterns if re.search(pattern, normalized, flags=re.I)
        )
        return placeholder_hits >= 1

    def _format_summary_with_meta(self, summary: str, target_language: str, video_title: str = None) -> str:
        """
        Add title and metadata to a summary.
        """
        language_name = self.language_map.get(target_language, "Chinese (Simplified)")
        meta_labels = self._get_summary_labels(target_language)
        
        # Do not add extra headings/disclaimers; keep the video title as an H1 when available.
        if video_title:
            prefix = f"# {video_title}\n\n"
        else:
            prefix = ""
        return prefix + summary

    def _generate_fallback_summary(self, transcript: str, target_language: str, video_title: str = None) -> str:
        """
        Generate a fallback summary when the OpenAI API is unavailable.
        
        Args:
            transcript: transcript text
            video_title: video title
            target_language: target language code
            
        Returns:
            Fallback summary text.
        """
        language_name = self.language_map.get(target_language, "Chinese (Simplified)")
        
        # Simple text processing to extract key information.
        lines = transcript.split('\n')
        content_lines = [line for line in lines if line.strip() and not line.startswith('#') and not line.startswith('**')]
        
        # Estimate content length.
        total_chars = sum(len(line) for line in content_lines)
        
        # Use labels for the target language.
        meta_labels = self._get_summary_labels(target_language)
        fallback_labels = self._get_fallback_labels(target_language)
        
        # Use the video title directly as the main title.
        title = video_title if video_title else "Summary"
        
        summary = f"""# {title}

**{meta_labels['language_label']}:** {language_name}
**{fallback_labels['notice']}:** {fallback_labels['api_unavailable']}



## {fallback_labels['overview_title']}

**{fallback_labels['content_length']}:** {fallback_labels['about']} {total_chars} {fallback_labels['characters']}
**{fallback_labels['paragraph_count']}:** {len(content_lines)} {fallback_labels['paragraphs']}

## {fallback_labels['main_content']}

{fallback_labels['content_description']}

{fallback_labels['suggestions_intro']}

1. {fallback_labels['suggestion_1']}
2. {fallback_labels['suggestion_2']}
3. {fallback_labels['suggestion_3']}

## {fallback_labels['recommendations']}

- {fallback_labels['recommendation_1']}
- {fallback_labels['recommendation_2']}


<br/>

<p style="color: #888; font-style: italic; text-align: center; margin-top: 16px;"><em>{fallback_labels['fallback_disclaimer']}</em></p>"""
        
        return summary
    
    def _get_current_time(self) -> str:
        """Get the current time string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_supported_languages(self) -> dict:
        """
        Get the supported language list.
        
        Returns:
            Mapping from language code to language name.
        """
        return self.language_map.copy()
    
    def _detect_transcript_language(self, transcript: str) -> str:
        """
        Detect the primary transcript language.
        
        Args:
            transcript: transcript text
            
        Returns:
            Detected language code.
        """
        # Simple language detection: look for language metadata in transcript text.
        legacy_detected_language_label = "**\u68c0\u6d4b\u8bed\u8a00:**"
        if "**Detected Language:**" in transcript or legacy_detected_language_label in transcript:
            # Extract detected language from Whisper transcript metadata.
            lines = transcript.split('\n')
            for line in lines:
                if "**Detected Language:**" in line or legacy_detected_language_label in line:
                    # Extract language code, for example: "**Detected Language:** en".
                    lang = line.split(":")[-1].strip()
                    return lang
        
        # If no language marker is found, use simple character detection.
        # Calculate the ratio of English letters, Chinese characters, and so on.
        total_chars = len(transcript)
        if total_chars == 0:
            return "en"  # Default to English.
            
        # Count Chinese characters.
        chinese_chars = sum(1 for char in transcript if '\u4e00' <= char <= '\u9fff')
        chinese_ratio = chinese_chars / total_chars
        
        # Count English letters.
        english_chars = sum(1 for char in transcript if char.isascii() and char.isalpha())
        english_ratio = english_chars / total_chars
        
        # Decide by ratio.
        if chinese_ratio > 0.3:
            return "zh"
        elif english_ratio > 0.3:
            return "en"
        else:
            return "en"  # Default to English.
    
    def _get_language_instruction(self, lang_code: str) -> str:
        """
        Get the language name used in optimization instructions by language code.
        
        Args:
            lang_code: language code
            
        Returns:
            Language name.
        """
        language_instructions = {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ar": "Arabic"
        }
        return language_instructions.get(lang_code, "English")
    

    def _get_summary_labels(self, lang_code: str) -> dict:
        """
        Get labels for the summary page.
        
        Args:
            lang_code: language code
            
        Returns:
            Label dictionary.
        """
        labels = {
            "en": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "zh": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "ja": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "ko": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "es": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "fr": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "de": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "it": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "pt": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "ru": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "ar": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            }
        }
        return labels.get(lang_code, labels["en"])
    
    def _get_fallback_labels(self, lang_code: str) -> dict:
        """
        Get labels for fallback summaries.
        
        Args:
            lang_code: language code
            
        Returns:
            Label dictionary.
        """
        labels = {
            "en": {
                "notice": "Notice",
                "api_unavailable": "OpenAI API is unavailable, this is a simplified summary",
                "overview_title": "Transcript Overview",
                "content_length": "Content Length",
                "about": "About",
                "characters": "characters",
                "paragraph_count": "Paragraph Count",
                "paragraphs": "paragraphs",
                "main_content": "Main Content",
                "content_description": "The transcript contains complete video speech content. Since AI summary cannot be generated currently, we recommend:",
                "suggestions_intro": "For detailed information, we suggest you:",
                "suggestion_1": "Review the complete transcript text for detailed information",
                "suggestion_2": "Focus on important paragraphs marked with timestamps",
                "suggestion_3": "Manually extract key points and takeaways",
                "recommendations": "Recommendations",
                "recommendation_1": "Configure OpenAI API key for better summary functionality",
                "recommendation_2": "Or use other AI services for text summarization",
                "fallback_disclaimer": "This is an automatically generated fallback summary"
            },
            "zh": {
                "notice": "Notice",
                "api_unavailable": "OpenAI API is unavailable, this is a simplified summary",
                "overview_title": "Transcript Overview",
                "content_length": "Content Length",
                "about": "About",
                "characters": "characters",
                "paragraph_count": "Paragraph Count",
                "paragraphs": "paragraphs",
                "main_content": "Main Content",
                "content_description": "The transcript contains complete video speech content. Since AI summary cannot be generated currently, we recommend:",
                "suggestions_intro": "For detailed information, we suggest you:",
                "suggestion_1": "Review the complete transcript text for detailed information",
                "suggestion_2": "Focus on important paragraphs marked with timestamps",
                "suggestion_3": "Manually extract key points and takeaways",
                "recommendations": "Recommendations",
                "recommendation_1": "Configure OpenAI API key for better summary functionality",
                "recommendation_2": "Or use other AI services for text summarization",
                "fallback_disclaimer": "This is an automatically generated fallback summary"
            }
        }
        return labels.get(lang_code, labels["en"])
    
    def is_available(self) -> bool:
        """
        Check whether the summary service is available.
        
        Returns:
            True if OpenAI API is configured, False otherwise
        """
        return self.client is not None
