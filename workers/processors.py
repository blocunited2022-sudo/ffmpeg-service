import os
import tempfile
import logging
import whisper
import asyncio
from uuid import UUID
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from app.models.task import TaskStatus
from app.services.supabase_service import supabase_service
from app.config import settings
from utils.file_utils import download_file, cleanup_temp_files, check_disk_space
from utils.ffmpeg_utils import (
    write_srt,
    write_ass,
    burn_subtitles,
    merge_video_audio,
    concat_videos,
    add_background_music
)

logger = logging.getLogger(__name__)

# Cache Whisper model globally to avoid reloading on every task
_whisper_model_cache: Optional[object] = None
_whisper_model_size: Optional[str] = None


def _load_whisper_model(model_size: str = "tiny"):
    """Load and cache Whisper model"""
    global _whisper_model_cache, _whisper_model_size

    if _whisper_model_cache is None or _whisper_model_size != model_size:
        logger.info(f"Loading Whisper model: {model_size}")
        os.environ["WHISPER_CACHE_DIR"] = settings.whisper_model_cache_dir
        import time
        start_time = time.time()
        _whisper_model_cache = whisper.load_model(model_size, download_root=settings.whisper_model_cache_dir)
        load_time = time.time() - start_time
        _whisper_model_size = model_size
        logger.info(f"Whisper model {model_size} loaded in {load_time:.2f}s")

    return _whisper_model_cache


async def process_caption_task(task_id: UUID, task_data: Dict[str, Any]) -> None:
    """
    Process a video captioning task

    Args:
        task_id: Task identifier
        task_data: Task data from Supabase
    """
    video_path = None
    output_path = None

    try:
        logger.info(f"[{task_id}] Starting caption task")
        logger.info(f"[{task_id}] Task data: {task_data}")

        logger.info(f"[{task_id}] Updating task status to RUNNING")
        supabase_service.update_task_status(task_id, TaskStatus.RUNNING)
        logger.info(f"[{task_id}] Status updated to RUNNING")

        video_url = task_data["video_url"]
        model_size = "base"

        video_filename = f"{task_id}_input.mp4"
        video_path = os.path.join(tempfile.gettempdir(), video_filename)

        if not check_disk_space(settings.max_file_size_bytes * 3):
            raise Exception("Insufficient disk space")

        logger.info(f"[{task_id}] Downloading video from {video_url}")
        _, file_size = await download_file(video_url, video_path)
        logger.info(f"[{task_id}] Video downloaded: {file_size/(1024*1024):.2f}MB")

        if not os.path.exists(video_path):
            raise Exception(f"Video file not found after download: {video_path}")

        logger.info(f"[{task_id}] Transcribing audio with Whisper model: {model_size}")

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            model = await loop.run_in_executor(executor, _load_whisper_model, model_size)
            logger.info(f"[{task_id}] Model ready, starting transcription...")
            import time
            transcribe_start = time.time()
            result = await loop.run_in_executor(
                executor,
                lambda: model.transcribe(
                    video_path,
                    fp16=False,
                    language="en",
                    verbose=False,
                    beam_size=1,
                    best_of=1
                )
            )
            transcribe_time = time.time() - transcribe_start
            logger.info(f"[{task_id}] Transcription took {transcribe_time:.2f}s")

        logger.info(f"[{task_id}] Transcription complete, found {len(result['segments'])} segments")
        subtitles = result["segments"]

        if len(subtitles) == 0:
            logger.warning(f"[{task_id}] No speech detected in video!")
        else:
            logger.info(f"[{task_id}] First subtitle: {subtitles[0].get('text', 'N/A')[:100]}...")

        # Define caption settings - BOLD, NO BLACK BOX, POSITIONED LOWER
        caption_settings = {
            "font-size": 70,              # Large size
            "primary-color": "#FFFFFF",    # White text
            "highlight-color": "#FFFF00",  # Yellow highlight
            "outline-color": "#000000",    # Black outline
            "shadow-color": "#000000",     # Black shadow
            "outline-width": 12,           # Thick outline for bold look
            "shadow-offset": 0,            # No shadow to avoid black box
            "max-words-per-line": 3,       # 3 words max per line
            "y": 1550,                     # Position (distance from top, 220 from bottom)
            "font-family": "Arial Black",  # Bold font
            "bold": True,                  # Bold enabled
            "highlight-position": "last",  # Highlight last word
            "use-ass": True                # Use ASS format
        }

        logger.info(f"[{task_id}] Generating ASS subtitles with custom styling")
        logger.info(f"[{task_id}] Caption settings: {caption_settings}")
        
        # Generate ASS format with settings (NO BLACK BOX)
        ass_text = write_ass(
            subtitles=subtitles,
            max_words_per_line=caption_settings["max-words-per-line"],
            settings=caption_settings
