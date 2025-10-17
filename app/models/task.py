from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum


class TaskType(str, Enum):
    """Enum for task types"""
    CAPTION = "caption"
    MERGE = "merge"
    BACKGROUND_MUSIC = "background_music"


class TaskStatus(str, Enum):
    """Enum for task status"""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class CaptionTaskRequest(BaseModel):
    """Request model for video captioning task"""
    video_url: HttpUrl = Field(..., description="URL of the video to add captions")
    model_size: str = Field(default="small", description="Whisper model size (tiny, base, small, medium, large)")

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "video_url": "https://example.com/video.mp4",
                "model_size": "small"
            }
        }
    }


class MergeTaskRequest(BaseModel):
    """Request model for video merging task"""
    scene_clip_urls: List[HttpUrl] = Field(..., min_length=1, description="List of scene video URLs")
    voiceover_urls: List[HttpUrl] = Field(..., min_length=1, description="List of voiceover audio URLs")
    width: int = Field(default=1080, ge=480, le=3840, description="Output video width")
    height: int = Field(default=1920, ge=480, le=3840, description="Output video height")
    video_volume: float = Field(default=0.2, ge=0.0, le=1.0, description="Volume level for video audio")
    voiceover_volume: float = Field(default=2.0, ge=0.0, le=10.0, description="Volume level for voiceover")

    model_config = {
        "json_schema_extra": {
            "example": {
                "scene_clip_urls": [
                    "https://example.com/scene1.mp4",
                    "https://example.com/scene2.mp4"
                ],
                "voiceover_urls": [
                    "https://example.com/voice1.mp3",
                    "https://example.com/voice2.mp3"
                ],
                "width": 1080,
                "height": 1920,
                "video_volume": 0.2,
                "voiceover_volume": 2.0
            }
        }
    }


class BackgroundMusicTaskRequest(BaseModel):
    """Request model for adding background music task"""
    video_url: HttpUrl = Field(..., description="URL of the video to add background music")
    music_url: HttpUrl = Field(..., description="URL of the background music file")
    music_volume: float = Field(default=0.3, ge=0.0, le=1.0, description="Volume level for background music")
    video_volume: float = Field(default=1.0, ge=0.0, le=1.0, description="Volume level for video audio")

    model_config = {
        "json_schema_extra": {
            "example": {
                "video_url": "https://example.com/video.mp4",
                "music_url": "https://example.com/music.mp3",
                "music_volume": 0.3,
                "video_volume": 1.0
            }
        }
    }


class TaskResponse(BaseModel):
    """Response model for task submission"""
    task_id: UUID = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    message: str = Field(..., description="Human-readable message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "message": "Task queued successfully"
            }
        }
    }


class TaskStatusResponse(BaseModel):
    """Response model for task status polling"""
    task_id: UUID
    status: TaskStatus
    video_url: Union[str, None] = Field(default=None, description="Public URL of processed video (if completed)")
    error: Union[str, None] = Field(default=None, description="Error message (if failed)")
    created_at: Union[datetime, None] = Field(default=None, description="Task creation timestamp")
    updated_at: Union[datetime, None] = Field(default=None, description="Last update timestamp")
    completed_at: Union[datetime, None] = Field(default=None, description="Task completion timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "success",
                "video_url": "https://confident-laughter-production-aec0.up.railway.app/video/550e8400-e29b-41d4-a716-446655440000_captioned.mp4",
                "error": None,
                "created_at": "2025-10-07T12:00:00Z",
                "updated_at": "2025-10-07T12:05:00Z",
                "completed_at": "2025-10-07T12:05:00Z"
            }
        }
    }


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Overall health status")
    redis: str = Field(..., description="Redis connection status")
    supabase: str = Field(..., description="Supabase connection status")
    queue_length: int = Field(..., description="Number of tasks in queue")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "redis": "connected",
                "supabase": "connected",
                "queue_length": 5
            }
        }
    }
