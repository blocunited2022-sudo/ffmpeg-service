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
        model_size = task_data.get("model_size", "base")
        
        logger.info(f"[{task_id}] Using Whisper model size: {model_size}")

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
            # Load the model with the correct size
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
            "y": 1550,                     # Position (distance from top, 370 from bottom)
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
        )
        
        logger.info(f"[{task_id}] ASS generation complete, length: {len(ass_text)} chars")
        logger.info(f"[{task_id}] ASS preview: {ass_text[:200]}...")

        output_filename = f"{task_id}_captioned.mp4"
        output_path = os.path.join(settings.video_output_dir, output_filename)

        logger.info(f"[{task_id}] Burning subtitles into video using FFmpeg")
        logger.info(f"[{task_id}] Input: {video_path}, Output: {output_path}")
        
        # Burn subtitles with the same settings
        with ThreadPoolExecutor() as burn_executor:
            await loop.run_in_executor(
                burn_executor,
                burn_subtitles,
                video_path,
                ass_text,
                output_path,
                caption_settings  # Pass settings to burn_subtitles
            )
        logger.info(f"[{task_id}] Subtitle burning complete")

        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"[{task_id}] Output video created: {output_size:.2f}MB")
        else:
            logger.error(f"[{task_id}] ERROR: Output video file not created!")

        result_url = f"{settings.railway_public_url}/video/{output_filename}"

        supabase_service.update_task_status(
            task_id,
            TaskStatus.SUCCESS,
            result_video_url=result_url,
            file_size=file_size
        )

        logger.info(f"[{task_id}] Caption task completed successfully")

    except Exception as e:
        error_msg = f"Caption task failed: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}", exc_info=True)
        supabase_service.update_task_status(
            task_id,
            TaskStatus.FAILED,
            error_message=error_msg
        )
        if output_path and os.path.exists(output_path):
            os.remove(output_path)

    finally:
        cleanup_temp_files(video_path)


async def process_merge_task(task_id: UUID, task_data: Dict[str, Any]) -> None:
    """
    Process a video merging task

    Args:
        task_id: Task identifier
        task_data: Task data from Supabase
    """
    temp_files = []
    scene_files = []
    concat_list_path = None
    output_path = None

    try:
        logger.info(f"[{task_id}] Starting merge task")
        logger.info(f"[{task_id}] Task data: {task_data}")

        logger.info(f"[{task_id}] Updating task status to RUNNING")
        supabase_service.update_task_status(task_id, TaskStatus.RUNNING)
        logger.info(f"[{task_id}] Status updated to RUNNING")

        metadata = task_data.get("metadata", {})
        scene_urls = metadata["scene_clip_urls"]
        voiceover_urls = metadata["voiceover_urls"]
        width = metadata.get("width", 1080)
        height = metadata.get("height", 1920)
        video_volume = metadata.get("video_volume", 0.2)
        voiceover_volume = metadata.get("voiceover_volume", 2.0)

        if not check_disk_space(settings.max_file_size_bytes * len(scene_urls) * 5):
            raise Exception("Insufficient disk space")

        temp_dir = tempfile.mkdtemp(prefix=f"merge_{task_id}_")
        total_size = 0

        for i, (scene_url, voice_url) in enumerate(zip(scene_urls, voiceover_urls)):
            logger.info(f"[{task_id}] Processing scene {i+1}/{len(scene_urls)}")

            scene_path = os.path.join(temp_dir, f"scene_{i}_video.mp4")
            voice_path = os.path.join(temp_dir, f"scene_{i}_audio.mp3")

            _, size1 = await download_file(scene_url, scene_path)
            _, size2 = await download_file(voice_url, voice_path)
            total_size += size1 + size2

            temp_files.extend([scene_path, voice_path])

            scene_output = os.path.join(temp_dir, f"scene_{i}_final.mp4")

            logger.info(f"[{task_id}] Merging scene {i+1} video and audio with FFmpeg")
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor,
                    merge_video_audio,
                    scene_path,
                    voice_path,
                    scene_output,
                    video_volume,
                    voiceover_volume,
                    5.0,
                    width,
                    height,
                    "cover"
                )
            logger.info(f"[{task_id}] Scene {i+1} merge complete")

            scene_files.append(scene_output)

        logger.info(f"[{task_id}] Concatenating {len(scene_files)} scenes")
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for scene_path in scene_files:
                escaped_path = scene_path.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        output_filename = f"{task_id}_merged.mp4"
        output_path = os.path.join(settings.video_output_dir, output_filename)

        logger.info(f"[{task_id}] Concatenating all scenes with FFmpeg")
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, concat_videos, concat_list_path, output_path)
        logger.info(f"[{task_id}] Concatenation complete")

        result_url = f"{settings.railway_public_url}/video/{output_filename}"

        supabase_service.update_task_status(
            task_id,
            TaskStatus.SUCCESS,
            result_video_url=result_url,
            file_size=total_size
        )

        logger.info(f"[{task_id}] Merge task completed successfully")

    except Exception as e:
        error_msg = f"Merge task failed: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}", exc_info=True)
        supabase_service.update_task_status(
            task_id,
            TaskStatus.FAILED,
            error_message=error_msg
        )
        if output_path and os.path.exists(output_path):
            os.remove(output_path)

    finally:
        cleanup_temp_files(*temp_files, *scene_files, concat_list_path)
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


async def process_background_music_task(task_id: UUID, task_data: Dict[str, Any]) -> None:
    """
    Process a background music task

    Args:
        task_id: Task identifier
        task_data: Task data from Supabase
    """
    video_path = None
    music_path = None
    output_path = None

    try:
        logger.info(f"[{task_id}] Starting background music task")
        logger.info(f"[{task_id}] Task data: {task_data}")

        logger.info(f"[{task_id}] Updating task status to RUNNING")
        supabase_service.update_task_status(task_id, TaskStatus.RUNNING)
        logger.info(f"[{task_id}] Status updated to RUNNING")

        video_url = task_data["video_url"]
        metadata = task_data.get("metadata", {})
        music_url = metadata["music_url"]
        music_volume = metadata.get("music_volume", 0.3)
        video_volume = metadata.get("video_volume", 1.0)

        if not check_disk_space(settings.max_file_size_bytes * 4):
            raise Exception("Insufficient disk space")

        temp_dir = tempfile.mkdtemp(prefix=f"music_{task_id}_")

        video_path = os.path.join(temp_dir, "input_video.mp4")
        music_path = os.path.join(temp_dir, "background_music.mp3")

        logger.info(f"[{task_id}] Downloading video and music")
        _, size1 = await download_file(video_url, video_path)
        _, size2 = await download_file(music_url, music_path)
        total_size = size1 + size2

        output_filename = f"{task_id}_with_music.mp4"
        output_path = os.path.join(settings.video_output_dir, output_filename)

        logger.info(f"[{task_id}] Adding background music to video with FFmpeg")
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                add_background_music,
                video_path,
                music_path,
                output_path,
                music_volume,
                video_volume
            )
        logger.info(f"[{task_id}] Background music addition complete")

        result_url = f"{settings.railway_public_url}/video/{output_filename}"

        supabase_service.update_task_status(
            task_id,
            TaskStatus.SUCCESS,
            result_video_url=result_url,
            file_size=total_size
        )

        logger.info(f"[{task_id}] Background music task completed successfully")

    except Exception as e:
        error_msg = f"Background music task failed: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}", exc_info=True)
        supabase_service.update_task_status(
            task_id,
            TaskStatus.FAILED,
            error_message=error_msg
        )
        if output_path and os.path.exists(output_path):
            os.remove(output_path)

    finally:
        cleanup_temp_files(video_path, music_path)
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
