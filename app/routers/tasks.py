from fastapi import APIRouter, HTTPException, status
from uuid import UUID
import logging
from app.models.task import (
    CaptionTaskRequest,
    MergeTaskRequest,
    BackgroundMusicTaskRequest,
    TaskResponse,
    TaskStatusResponse,
    TaskType,
    TaskStatus
)
from app.services.redis_service import redis_service
from app.services.supabase_service import supabase_service
from app.config import settings
from utils.file_utils import check_file_size, FileSizeLimitExceeded, DownloadError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("/caption", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_caption_task(request: CaptionTaskRequest):
    """
    Submit a video captioning task

    This endpoint queues a video for caption generation using OpenAI Whisper.
    Returns immediately with a task_id that can be used to poll for status.

    - **video_url**: URL of the video to process (max 100MB)
    - **model_size**: Whisper model size (tiny, base, small, medium, large)
    """
    try:
        video_url_str = str(request.video_url)

        try:
            await check_file_size(video_url_str)
        except FileSizeLimitExceeded as e:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=str(e)
            )
        except DownloadError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to access video URL: {str(e)}"
            )

        task_id = supabase_service.create_task(
            task_type=TaskType.CAPTION,
            video_url=video_url_str,
            model_size=request.model_size
        )

        if not task_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create task in database"
            )

        enqueued = await redis_service.enqueue_task(task_id, TaskType.CAPTION.value)

        if not enqueued:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enqueue task"
            )

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            message="Caption task queued successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating caption task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/merge", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_merge_task(request: MergeTaskRequest):
    """
    Submit a video merging task

    This endpoint queues multiple scene videos and voiceovers to be merged.
    Returns immediately with a task_id that can be used to poll for status.

    - **scene_clip_urls**: List of scene video URLs
    - **voiceover_urls**: List of voiceover audio URLs (must match scene count)
    - **width**: Output video width (default: 1080)
    - **height**: Output video height (default: 1920)
    - **video_volume**: Volume for video audio (0.0-1.0, default: 0.2)
    - **voiceover_volume**: Volume for voiceover (0.0-10.0, default: 2.0)
    """
    try:
        if len(request.scene_clip_urls) != len(request.voiceover_urls):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of scene clips must match number of voiceovers"
            )

        scene_urls = [str(url) for url in request.scene_clip_urls]
        voiceover_urls = [str(url) for url in request.voiceover_urls]

        total_size = 0
        for url in scene_urls + voiceover_urls:
            try:
                size = await check_file_size(url)
                total_size += size
            except FileSizeLimitExceeded as e:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=str(e)
                )
            except DownloadError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unable to access URL {url}: {str(e)}"
                )

        max_total_size = settings.max_file_size_mb * 5 * 1024 * 1024
        if total_size > max_total_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Total file size {total_size/(1024*1024):.2f}MB exceeds limit of {settings.max_file_size_mb * 5}MB"
            )

        metadata = {
            "scene_clip_urls": scene_urls,
            "voiceover_urls": voiceover_urls,
            "width": request.width,
            "height": request.height,
            "video_volume": request.video_volume,
            "voiceover_volume": request.voiceover_volume
        }

        task_id = supabase_service.create_task(
            task_type=TaskType.MERGE,
            video_url=scene_urls[0],
            metadata=metadata
        )

        if not task_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create task in database"
            )

        enqueued = await redis_service.enqueue_task(task_id, TaskType.MERGE.value)

        if not enqueued:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enqueue task"
            )

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            message="Merge task queued successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating merge task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/background-music", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_background_music_task(request: BackgroundMusicTaskRequest):
    """
    Submit a background music task

    This endpoint queues a video to have background music added.
    Returns immediately with a task_id that can be used to poll for status.

    - **video_url**: URL of the video to process
    - **music_url**: URL of the background music file
    - **music_volume**: Volume for background music (0.0-1.0, default: 0.3)
    - **video_volume**: Volume for video audio (0.0-1.0, default: 1.0)
    """
    try:
        video_url_str = str(request.video_url)
        music_url_str = str(request.music_url)

        total_size = 0
        for url in [video_url_str, music_url_str]:
            try:
                size = await check_file_size(url)
                total_size += size
            except FileSizeLimitExceeded as e:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=str(e)
                )
            except DownloadError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unable to access URL {url}: {str(e)}"
                )

        max_total_size = settings.max_file_size_mb * 2 * 1024 * 1024
        if total_size > max_total_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Total file size {total_size/(1024*1024):.2f}MB exceeds limit of {settings.max_file_size_mb * 2}MB"
            )

        metadata = {
            "music_url": music_url_str,
            "music_volume": request.music_volume,
            "video_volume": request.video_volume
        }

        task_id = supabase_service.create_task(
            task_type=TaskType.BACKGROUND_MUSIC,
            video_url=video_url_str,
            metadata=metadata
        )

        if not task_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create task in database"
            )

        enqueued = await redis_service.enqueue_task(task_id, TaskType.BACKGROUND_MUSIC.value)

        if not enqueued:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enqueue task"
            )

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            message="Background music task queued successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating background music task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: UUID):
    """
    Get the status of a task

    Poll this endpoint to check the status of your video processing task.

    Returns:
    - **status**: queued, running, success, or failed
    - **video_url**: Public URL of processed video (only when status is success)
    - **error**: Error message (only when status is failed)
    """
    try:
        logger.info(f"Fetching status for task {task_id}")
        task_data = supabase_service.get_task(task_id)

        if not task_data:
            logger.warning(f"Task {task_id} not found in database")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )

        logger.info(f"Task {task_id} found with status: {task_data['status']}")

        return TaskStatusResponse(
            task_id=UUID(task_data["id"]),
            status=TaskStatus(task_data["status"]),
            video_url=task_data.get("result_video_url"),
            error=task_data.get("error_message"),
            created_at=task_data.get("created_at"),
            updated_at=task_data.get("updated_at"),
            completed_at=task_data.get("completed_at")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
