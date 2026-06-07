from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional
import uuid
import json
import re
import sqlite3
import threading
import time
from urllib.parse import quote
import openai

from video_processor import VideoProcessor
from transcriber import Transcriber
from summarizer import Summarizer
from translator import Translator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI视频转录器", version="1.0.0")

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

# 创建临时目录
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# 初始化处理器
video_processor = VideoProcessor()
transcriber = Transcriber()
summarizer = Summarizer()
translator = Translator()

TASKS_FILE = TEMP_DIR / "tasks.json"
DB_FILE = TEMP_DIR / "artifacts.sqlite3"
tasks_lock = threading.Lock()


def _db_connect():
    conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化 SQLite 存储：任务状态和所有生成的 Markdown 工件。"""
    with tasks_lock:
        with _db_connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    filename TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_updated ON tasks(updated_at)")
            conn.commit()

def load_tasks():
    """加载任务状态"""
    init_db()
    try:
        with _db_connect() as conn:
            rows = conn.execute("SELECT task_id, data FROM tasks").fetchall()
        if rows:
            return {row["task_id"]: json.loads(row["data"]) for row in rows}

        if TASKS_FILE.exists():
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                legacy_tasks = json.load(f)
            save_tasks(legacy_tasks)
            migrate_task_artifacts(legacy_tasks)
            return legacy_tasks
    except Exception as e:
        logger.error(f"加载任务状态失败: {e}")
    return {}

def save_tasks(tasks_data):
    """保存任务状态"""
    try:
        with tasks_lock:
            now = time.time()
            with _db_connect() as conn:
                known_ids = set()
                for task_id, task_data in tasks_data.items():
                    known_ids.add(task_id)
                    existing = conn.execute(
                        "SELECT created_at FROM tasks WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
                    created_at = existing["created_at"] if existing else now
                    conn.execute(
                        """
                        INSERT INTO tasks(task_id, data, created_at, updated_at)
                        VALUES(?, ?, ?, ?)
                        ON CONFLICT(task_id) DO UPDATE SET
                            data = excluded.data,
                            updated_at = excluded.updated_at
                        """,
                        (
                            task_id,
                            json.dumps(task_data, ensure_ascii=False),
                            created_at,
                            now,
                        ),
                    )
                if known_ids:
                    placeholders = ",".join("?" for _ in known_ids)
                    conn.execute(
                        f"DELETE FROM tasks WHERE task_id NOT IN ({placeholders})",
                        tuple(known_ids),
                    )
                else:
                    conn.execute("DELETE FROM tasks")
                conn.commit()
    except Exception as e:
        logger.error(f"保存任务状态失败: {e}")


def save_artifact(task_id: str, artifact_type: str, filename: str, content: str) -> None:
    """保存生成工件到 SQLite。"""
    with tasks_lock:
        with _db_connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts(task_id, artifact_type, filename, content, created_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    task_id = excluded.task_id,
                    artifact_type = excluded.artifact_type,
                    content = excluded.content
                """,
                (task_id, artifact_type, filename, content, time.time()),
            )
            conn.commit()


def _legacy_artifact_content(task_data: dict, content_key: str, path_key: str, filename: str) -> Optional[str]:
    content = task_data.get(content_key)
    if content:
        return content
    raw_path = task_data.get(path_key)
    candidate = Path(raw_path) if raw_path else TEMP_DIR / filename
    try:
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"迁移旧工件失败 {candidate}: {e}")
    return None


def migrate_task_artifacts(tasks_data: dict) -> None:
    """把旧 task JSON 中内联保存的 Markdown 内容迁移到 SQLite artifacts。"""
    for task_id, task_data in tasks_data.items():
        if task_data.get("status") != "completed":
            continue
        safe_title = task_data.get("safe_title") or "untitled"
        short_id = task_data.get("short_id") or task_id.replace("-", "")[:6]
        raw_filename = task_data.get("raw_script_file")
        script_filename = (
            task_data.get("script_filename")
            or Path(task_data.get("script_path", "")).name
            or f"transcript_{safe_title}_{short_id}.md"
        )
        summary_filename = (
            task_data.get("summary_filename")
            or Path(task_data.get("summary_path", "")).name
            or f"summary_{safe_title}_{short_id}.md"
        )
        translation_filename = (
            task_data.get("translation_filename")
            or Path(task_data.get("translation_path", "")).name
            or f"translation_{safe_title}_{short_id}.md"
        )
        artifact_specs = [
            (
                "raw",
                raw_filename,
                _legacy_artifact_content(task_data, "raw_script", "raw_script_path", raw_filename)
                if raw_filename else None,
            ),
            (
                "script",
                script_filename,
                _legacy_artifact_content(task_data, "script", "script_path", script_filename),
            ),
            (
                "summary",
                summary_filename,
                _legacy_artifact_content(task_data, "summary", "summary_path", summary_filename),
            ),
            (
                "translation",
                translation_filename,
                _legacy_artifact_content(task_data, "translation", "translation_path", translation_filename),
            ),
        ]
        for artifact_type, filename, content in artifact_specs:
            if filename and content:
                save_artifact(task_id, artifact_type, filename, content)


def get_artifact_by_filename(filename: str) -> Optional[sqlite3.Row]:
    with _db_connect() as conn:
        return conn.execute(
            "SELECT filename, content FROM artifacts WHERE filename = ?",
            (filename,),
        ).fetchone()


def delete_task_artifacts(task_id: str) -> None:
    with tasks_lock:
        with _db_connect() as conn:
            conn.execute("DELETE FROM artifacts WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            conn.commit()

async def broadcast_task_update(task_id: str, task_data: dict):
    """向所有连接的SSE客户端广播任务状态更新"""
    logger.info(f"广播任务更新: {task_id}, 状态: {task_data.get('status')}, 连接数: {len(sse_connections.get(task_id, []))}")
    if task_id in sse_connections:
        connections_to_remove = []
        for queue in sse_connections[task_id]:
            try:
                await queue.put(json.dumps(task_data, ensure_ascii=False))
                logger.debug(f"消息已发送到队列: {task_id}")
            except Exception as e:
                logger.warning(f"发送消息到队列失败: {e}")
                connections_to_remove.append(queue)
        
        # 移除断开的连接
        for queue in connections_to_remove:
            sse_connections[task_id].remove(queue)
        
        # 如果没有连接了，清理该任务的连接列表
        if not sse_connections[task_id]:
            del sse_connections[task_id]

# 启动时加载任务状态
tasks = load_tasks()
# 存储正在处理的URL，防止重复处理
processing_urls = set()
# 存储活跃的任务对象，用于控制和取消
active_tasks = {}
# 存储SSE连接，用于实时推送状态更新
sse_connections = {}

# 本地上传：允许的类型
UPLOAD_ALLOWED_EXT = frozenset({".txt", ".mp3", ".mp4", ".m4a", ".wav", ".webm", ".mkv", ".ogg", ".flac"})


def _sanitize_title_for_filename(title: str) -> str:
    """将视频标题清洗为安全的文件名片段。"""
    if not title:
        return "untitled"
    # 仅保留字母数字、下划线、连字符与空格
    safe = re.sub(r"[^\w\-\s]", "", title)
    # 压缩空白并转为下划线
    safe = re.sub(r"\s+", "_", safe).strip("._-")
    # 最长限制，避免过长文件名问题
    return safe[:80] or "untitled"


def _txt_to_raw_transcript_markdown(body: str) -> str:
    """将纯文本包装为与 Whisper 输出结构一致的 Markdown。"""
    text = body.strip() if body.strip() else "(empty)"
    return "\n".join([
        "# Video Transcription",
        "",
        "**Detected Language:**",
        "**Language Probability:** —",
        "",
        "## Transcription Content",
        "",
        text,
    ])


def _format_timestamp(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _count_text_units(text: Optional[str]) -> dict:
    body = text or ""
    return {
        "chars": len(body),
        "words": len(re.findall(r"[A-Za-z0-9_]+|[\u0400-\u04ff]+|[\u4e00-\u9fff]", body)),
        "lines": len([line for line in body.splitlines() if line.strip()]),
    }


def _build_task_statistics(
    task_data: dict,
    *,
    video_title: str,
    source_ref: str,
    extraction_method: str,
    detected_language: str,
    summary_language: str,
    raw_script: str,
    script: str,
    summary: str,
    translation: Optional[str],
    model_id: str,
) -> dict:
    finished_at = time.time()
    try:
        started_at = float(task_data.get("processing_started_at") or finished_at)
    except (TypeError, ValueError):
        started_at = finished_at
    elapsed_seconds = max(0.0, finished_at - started_at)

    return {
        "processing_started_at": _format_timestamp(started_at),
        "processing_finished_at": _format_timestamp(finished_at),
        "processing_seconds": round(elapsed_seconds, 2),
        "processing_minutes": round(elapsed_seconds / 60, 2),
        "input_type": task_data.get("input_type") or ("upload" if source_ref.startswith("upload:") else "url"),
        "input_name": task_data.get("input_name") or source_ref,
        "source_ref": source_ref,
        "video_title": video_title,
        "extraction_method": extraction_method,
        "detected_language": detected_language,
        "summary_language": summary_language,
        "translation_generated": bool(translation),
        "model": (model_id or "").strip() or "server default",
        "raw_transcript": _count_text_units(raw_script),
        "optimized_transcript": _count_text_units(script),
        "summary": _count_text_units(summary),
        "translation": _count_text_units(translation),
    }


async def _run_post_extract_pipeline(
    task_id: str,
    raw_script: str,
    video_title: str,
    source_ref: str,
    summary_language: str,
    request_summarizer: Summarizer,
    extraction_method: str,
    dedup_url: Optional[str] = None,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
) -> None:
    """取得 raw_script 后的共用管线：归档、优化、翻译、摘要、广播。"""
    short_id = task_id.replace("-", "")[:6]
    safe_title = _sanitize_title_for_filename(video_title)

    try:
        raw_md_filename = f"raw_{safe_title}_{short_id}.md"
        save_artifact(
            task_id,
            "raw",
            raw_md_filename,
            (raw_script or "") + f"\n\nsource: {source_ref}\n",
        )
        tasks[task_id].update({"raw_script_file": raw_md_filename})
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
    except Exception as e:
        logger.error(f"保存原始转录Markdown失败: {e}")

    tasks[task_id].update({
        "progress": 55,
        "message": "正在优化转录文本...",
    })
    save_tasks(tasks)
    await broadcast_task_update(task_id, tasks[task_id])

    script = await request_summarizer.optimize_transcript(raw_script)

    script_with_title = f"# {video_title}\n\n{script}\n\nsource: {source_ref}\n"

    detected_language = transcriber.get_detected_language(raw_script)
    detected_language = (detected_language or "").strip()
    if not detected_language:
        detected_language = translator.infer_language_code(raw_script)
    detected_language = translator.normalize_lang_code(detected_language) or detected_language

    logger.info(f"检测到的语言: {detected_language}, 摘要语言: {summary_language}")

    translation_content = None
    translation_filename = None

    eff_key = (api_key or "").strip()
    eff_base = (model_base_url or "").strip().rstrip("/")
    if eff_key:
        request_translator = Translator(
            api_key=eff_key,
            base_url=eff_base or None,
            model=model_id or None,
        )
    else:
        request_translator = translator

    need_translation = translator.languages_differ_for_translation(
        detected_language, summary_language
    )

    if need_translation:
        logger.info(f"需要翻译: {detected_language} -> {summary_language}")
        tasks[task_id].update({
            "progress": 70,
            "message": "正在生成翻译...",
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

        translation_content = await request_translator.translate_text(
            script, summary_language, detected_language
        )
        translation_with_title = f"# {video_title}\n\n{translation_content}\n\nsource: {source_ref}\n"
        translation_filename = f"translation_{safe_title}_{short_id}.md"
        save_artifact(task_id, "translation", translation_filename, translation_with_title)
    else:
        logger.info(
            f"不需要翻译: detected_language={detected_language}, summary_language={summary_language}, "
            f"need_translation={need_translation}"
        )

    tasks[task_id].update({
        "progress": 80,
        "message": "正在生成摘要...",
    })
    save_tasks(tasks)
    await broadcast_task_update(task_id, tasks[task_id])

    summary = await request_summarizer.summarize(script, summary_language, video_title)
    summary_with_source = summary + f"\n\nsource: {source_ref}\n"
    statistics = _build_task_statistics(
        tasks[task_id],
        video_title=video_title,
        source_ref=source_ref,
        extraction_method=extraction_method,
        detected_language=detected_language,
        summary_language=summary_language,
        raw_script=raw_script,
        script=script,
        summary=summary,
        translation=translation_content,
        model_id=model_id,
    )

    script_filename = f"transcript_{safe_title}_{short_id}.md"
    save_artifact(task_id, "script", script_filename, script_with_title)

    summary_filename = f"summary_{safe_title}_{short_id}.md"
    save_artifact(task_id, "summary", summary_filename, summary_with_source)

    task_result = {
        "status": "completed",
        "progress": 100,
        "message": "处理完成！",
        "video_title": video_title,
        "script": script_with_title,
        "summary": summary_with_source,
        "script_filename": script_filename,
        "summary_filename": summary_filename,
        "short_id": short_id,
        "safe_title": safe_title,
        "detected_language": detected_language,
        "summary_language": summary_language,
        "statistics": statistics,
    }

    if translation_content and translation_filename:
        task_result.update({
            "translation": translation_with_title,
            "translation_filename": translation_filename,
        })

    tasks[task_id].update(task_result)
    save_tasks(tasks)
    logger.info(f"任务完成，准备广播最终状态: {task_id}")
    await broadcast_task_update(task_id, tasks[task_id])
    logger.info(f"最终状态已广播: {task_id}")

    if dedup_url:
        processing_urls.discard(dedup_url)
    if task_id in active_tasks:
        del active_tasks[task_id]


@app.get("/")
async def read_root():
    """返回前端页面"""
    return FileResponse(str(PROJECT_ROOT / "static" / "index.html"))

@app.post("/api/models")
async def list_models(
    base_url: str = Form(default=""),
    api_key:  str = Form(default=""),
):
    """Proxy: fetch model list from any OpenAI-compatible API."""
    effective_key = api_key or os.getenv("OPENAI_API_KEY", "")
    effective_url = base_url.rstrip("/") or os.getenv("OPENAI_BASE_URL") or None

    if not effective_key:
        raise HTTPException(status_code=400, detail="API key is required")

    try:
        client = openai.OpenAI(api_key=effective_key, base_url=effective_url)
        resp   = await asyncio.to_thread(client.models.list)
        models = [{"id": m.id, "name": getattr(m, "name", m.id)} for m in resp.data]
        # Sort by id for readability
        models.sort(key=lambda x: x["id"])
        return {"data": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _enqueue_upload_job(
    file: UploadFile,
    summary_language: str,
    api_key: str,
    model_base_url: str,
    model_id: str,
) -> dict:
    """保存上传文件并入队 process_upload_task，返回 {task_id, message}。"""
    raw_name = file.filename or "upload.bin"
    if ".." in raw_name or "/" in raw_name or "\\" in raw_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    safe_name = os.path.basename(raw_name)
    ext = Path(safe_name).suffix.lower()
    if ext not in UPLOAD_ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext or '(none)'}",
        )

    task_id = str(uuid.uuid4())
    unique_stem = task_id.replace("-", "")[:12]
    dest = TEMP_DIR / f"upload_{unique_stem}{ext}"

    total = 0
    with open(dest, "wb") as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            out_f.write(chunk)

    if total == 0:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Empty file")

    video_title = _sanitize_title_for_filename(Path(safe_name).stem) or "upload"
    source_label = f"upload:{safe_name}"

    started_at = time.time()
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "message": "开始处理上传文件...",
        "script": None,
        "summary": None,
        "error": None,
        "url": source_label,
        "input_name": safe_name,
        "input_type": "upload",
        "upload_ext": ext,
        "summary_language": summary_language,
        "model_id": model_id or "",
        "processing_started_at": started_at,
    }
    save_tasks(tasks)

    bg = asyncio.create_task(
        process_upload_task(
            task_id,
            dest,
            safe_name,
            video_title,
            ext,
            summary_language,
            api_key,
            model_base_url,
            model_id,
        )
    )
    active_tasks[task_id] = bg

    return {"task_id": task_id, "message": "任务已创建，正在处理中..."}


@app.post("/api/process-video")
async def process_video(
    url: str = Form(default=""),
    summary_language: str = Form(default="zh"),
    api_key: str = Form(default=""),
    model_base_url: str = Form(default=""),
    model_id: str = Form(default=""),
    file: Optional[UploadFile] = File(None),
):
    """
    处理视频链接或本地上传（multipart 中带 file 且无有效 URL 时走上传流程）。
    上传与 URL 共用此路径，便于反向代理只放行 /api/process-video 的环境。
    """
    try:
        if file is not None and (file.filename or "").strip():
            return await _enqueue_upload_job(
                file, summary_language, api_key, model_base_url, model_id
            )

        stripped = (url or "").strip()
        if not stripped:
            raise HTTPException(
                status_code=400,
                detail="Provide a video URL or upload a file",
            )

        url = stripped

        # 检查是否已经在处理相同的URL
        if url in processing_urls:
            # 查找现有任务
            for tid, task in tasks.items():
                if task.get("url") == url:
                    return {"task_id": tid, "message": "该视频正在处理中，请等待..."}
            
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        
        # 标记URL为正在处理
        processing_urls.add(url)
        
        # 初始化任务状态
        started_at = time.time()
        tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "开始处理视频...",
            "script": None,
            "summary": None,
            "error": None,
            "url": url,  # 保存URL用于去重
            "input_name": url,
            "input_type": "url",
            "summary_language": summary_language,
            "model_id": model_id or "",
            "processing_started_at": started_at,
        }
        save_tasks(tasks)
        
        # 创建并跟踪异步任务
        task = asyncio.create_task(process_video_task(task_id, url, summary_language, api_key, model_base_url, model_id))
        active_tasks[task_id] = task
        
        return {"task_id": task_id, "message": "任务已创建，正在处理中..."}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理视频时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

async def process_video_task(
    task_id: str,
    url: str,
    summary_language: str,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
):
    """
    异步处理视频任务
    """
    try:
        # ── 阶段一：优先尝试获取平台字幕（快速路径） ──────────────────────
        tasks[task_id].update({
            "status": "processing",
            "progress": 10,
            "message": "正在检测视频字幕..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
        await asyncio.sleep(0.1)

        # 如果前端传入了 API 凭据，创建专用 Summarizer（线程安全，覆盖全局实例）
        if api_key:
            effective_url = model_base_url.rstrip("/") or None
            request_summarizer = Summarizer(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            logger.info(f"使用前端提供的 API Key，base_url={effective_url}, model={model_id or 'default'}")
        else:
            request_summarizer = summarizer  # 全局实例（使用环境变量）

        subtitle_text, sub_title, sub_lang = await video_processor.fetch_subtitles(url, TEMP_DIR)

        if subtitle_text:
            # ── 快速路径：有字幕，跳过音频下载和 Whisper ──────────────────
            video_title = sub_title
            raw_script = subtitle_text
            extraction_method = "subtitle"
            # 把语言写入 transcriber，保持下游逻辑一致
            transcriber.last_detected_language = sub_lang

            tasks[task_id].update({
                "progress": 40,
                "message": f"字幕获取成功（{sub_lang}），正在处理文本..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])
        else:
            # ── 慢速路径：无字幕，下载音频 → Whisper 转录 ─────────────────
            tasks[task_id].update({
                "progress": 15,
                "message": "未找到字幕，正在下载视频音频..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            audio_path, video_title = await video_processor.download_and_convert(
                url, TEMP_DIR, prefetched_title=sub_title or None
            )

            tasks[task_id].update({
                "progress": 35,
                "message": "音频下载完成，准备转录..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            tasks[task_id].update({
                "progress": 40,
                "message": "正在转录音频（Whisper）..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            raw_script = await transcriber.transcribe(audio_path)
            extraction_method = "whisper"

        await _run_post_extract_pipeline(
            task_id=task_id,
            raw_script=raw_script,
            video_title=video_title,
            source_ref=url,
            summary_language=summary_language,
            request_summarizer=request_summarizer,
            extraction_method=extraction_method,
            dedup_url=url,
            api_key=api_key,
            model_base_url=model_base_url,
            model_id=model_id,
        )

        # 不要立即删除临时文件！保留给用户下载
        # 文件会在一定时间后自动清理或用户手动清理

    except Exception as e:
        logger.error(f"任务 {task_id} 处理失败: {str(e)}")
        # 从处理列表中移除URL
        processing_urls.discard(url)
        
        # 从活跃任务列表中移除
        if task_id in active_tasks:
            del active_tasks[task_id]
            
        tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "message": f"处理失败: {str(e)}"
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

@app.post("/api/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    summary_language: str = Form(default="zh"),
    api_key: str = Form(default=""),
    model_base_url: str = Form(default=""),
    model_id: str = Form(default=""),
):
    """独立上传入口；逻辑与 multipart 带 file 的 /api/process-video 相同。"""
    return await _enqueue_upload_job(
        file, summary_language, api_key, model_base_url, model_id
    )


async def process_upload_task(
    task_id: str,
    saved_path: Path,
    original_name: str,
    video_title: str,
    ext_lower: str,
    summary_language: str,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
):
    source_ref = f"upload:{original_name}"
    try:
        if api_key:
            effective_url = model_base_url.rstrip("/") or None
            request_summarizer = Summarizer(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            logger.info(
                f"上传任务使用前端 API Key，base_url={effective_url}, model={model_id or 'default'}"
            )
        else:
            request_summarizer = summarizer

        if ext_lower == ".txt":
            extraction_method = "text_upload"
            tasks[task_id].update({
                "progress": 20,
                "message": "正在读取文本文件...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            body = saved_path.read_text(encoding="utf-8", errors="replace")
            if not body.strip():
                raise Exception("文本文件为空")
            transcriber.last_detected_language = None
            raw_script = _txt_to_raw_transcript_markdown(body)
        else:
            tasks[task_id].update({
                "progress": 15,
                "message": "正在转换音频格式...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            audio_path = await video_processor.normalize_local_media_to_m4a(saved_path, TEMP_DIR)

            tasks[task_id].update({
                "progress": 35,
                "message": "音频准备完成，准备转录...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            tasks[task_id].update({
                "progress": 40,
                "message": "正在转录音频（Whisper）...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            raw_script = await transcriber.transcribe(audio_path)
            extraction_method = "whisper_upload"

        await _run_post_extract_pipeline(
            task_id=task_id,
            raw_script=raw_script,
            video_title=video_title,
            source_ref=source_ref,
            summary_language=summary_language,
            request_summarizer=request_summarizer,
            extraction_method=extraction_method,
            dedup_url=None,
            api_key=api_key,
            model_base_url=model_base_url,
            model_id=model_id,
        )

    except Exception as e:
        logger.error(f"任务 {task_id} 处理失败: {str(e)}")
        if task_id in active_tasks:
            del active_tasks[task_id]
        tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "message": f"处理失败: {str(e)}",
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])


@app.get("/api/artifacts")
async def list_artifacts():
    """返回已完成任务列表，供前端按输入文件名打开历史结果。"""
    items = []
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT task_id, data, updated_at
            FROM tasks
            ORDER BY updated_at DESC
            LIMIT 100
            """
        ).fetchall()

    for row in rows:
        try:
            task = json.loads(row["data"])
        except Exception:
            continue
        if task.get("status") != "completed":
            continue
        items.append({
            "task_id": row["task_id"],
            "input_name": task.get("input_name") or task.get("video_title") or task.get("url") or row["task_id"],
            "video_title": task.get("video_title"),
            "updated_at": row["updated_at"],
            "has_translation": bool(task.get("translation")),
        })
    return {"items": items}


@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks[task_id]

@app.get("/api/task-stream/{task_id}")
async def task_stream(task_id: str):
    """
    SSE实时任务状态流
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    async def event_generator():
        # 创建任务专用的队列
        queue = asyncio.Queue()
        
        # 将队列添加到连接列表
        if task_id not in sse_connections:
            sse_connections[task_id] = []
        sse_connections[task_id].append(queue)
        
        try:
            # 立即发送当前状态
            current_task = tasks.get(task_id, {})
            yield f"data: {json.dumps(current_task, ensure_ascii=False)}\n\n"
            
            # 持续监听状态更新
            while True:
                try:
                    # 等待状态更新，超时时间30秒发送心跳
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {data}\n\n"
                    
                    # 如果任务完成或失败，结束流
                    task_data = json.loads(data)
                    if task_data.get("status") in ["completed", "error"]:
                        break
                        
                except asyncio.TimeoutError:
                    # 发送心跳保持连接
                    yield f"data: {json.dumps({'type': 'heartbeat'}, ensure_ascii=False)}\n\n"
                    
        except asyncio.CancelledError:
            logger.info(f"SSE连接被取消: {task_id}")
        except Exception as e:
            logger.error(f"SSE流异常: {e}")
        finally:
            # 清理连接
            if task_id in sse_connections and queue in sse_connections[task_id]:
                sse_connections[task_id].remove(queue)
                if not sse_connections[task_id]:
                    del sse_connections[task_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """
    从 SQLite 下载生成的 Markdown 工件。
    """
    try:
        # 检查文件扩展名安全性
        if not filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="仅支持下载.md文件")
        
        # 检查文件名格式（防止路径遍历攻击）
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="文件名格式无效")
            
        artifact = get_artifact_by_filename(filename)
        if not artifact:
            raise HTTPException(status_code=404, detail="文件不存在")
            
        quoted = quote(filename)
        return Response(
            content=artifact["content"],
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"; filename*=UTF-8\'\'{quoted}'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """
    取消并删除任务
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 如果任务还在运行，先取消它
    if task_id in active_tasks:
        task = active_tasks[task_id]
        if not task.done():
            task.cancel()
            logger.info(f"任务 {task_id} 已被取消")
        del active_tasks[task_id]
    
    # 从处理URL列表中移除
    task_url = tasks[task_id].get("url")
    if task_url:
        processing_urls.discard(task_url)
    
    # 删除任务记录和 SQLite 工件
    del tasks[task_id]
    delete_task_artifacts(task_id)
    return {"message": "任务已取消并删除"}

@app.get("/api/tasks/active")
async def get_active_tasks():
    """
    获取当前活跃任务列表（用于调试）
    """
    active_count = len(active_tasks)
    processing_count = len(processing_urls)
    return {
        "active_tasks": active_count,
        "processing_urls": processing_count,
        "task_ids": list(active_tasks.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099)
