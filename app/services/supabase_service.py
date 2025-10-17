import logging
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from supabase import create_client, Client
from app.config import settings
from app.models.task import TaskType, TaskStatus

logger = logging.getLogger(__name__)


class SupabaseService:
    """Service for managing task data in Supabase"""

    def __init__(self):
        self.client: Optional[Client] = None

    def connect(self) -> None:
        """Establish connection to Supabase"""
        try:
            logger.info(f"Attempting Supabase connection with URL: {settings.supabase_url}")
            logger.info(f"Supabase key present: {bool(settings.supabase_key)}")

            if not settings.supabase_url or not settings.supabase_key:
                logger.warning("Supabase credentials not configured")
                return

            self.client = create_client(settings.supabase_url, settings.supabase_key)
            logger.info("Supabase connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise

    def is_healthy(self) -> bool:
        """Check if Supabase connection is healthy"""
        try:
            if not self.client:
                return False
            result = self.client.table("tasks").select("id").limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Supabase health check failed: {e}")
            return False

    def create_task(
        self,
        task_type: TaskType,
        video_url: str,
        model_size: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[UUID]:
        """
        Create a new task record in Supabase

        Args:
            task_type: Type of task
            video_url: Input video URL
            model_size: Whisper model size (for caption tasks)
            metadata: Additional task-specific data

        Returns:
            Task UUID if created successfully, None otherwise
        """
        try:
            task_data = {
                "task_type": task_type.value,
                "status": TaskStatus.QUEUED.value,
                "video_url": video_url,
                "model_size": model_size,
                "metadata": metadata or {}
            }

            logger.debug(f"Creating task with data: {task_data}")
            result = self.client.table("tasks").insert(task_data).execute()

            if result.data and len(result.data) > 0:
                task_id = UUID(result.data[0]["id"])
                logger.info(f"Created task {task_id} with type {task_type.value}")
                logger.debug(f"Full task data: {result.data[0]}")
                return task_id
            logger.error(f"Failed to create task - no data returned from insert")
            return None
        except Exception as e:
            logger.error(f"Failed to create task: {e}", exc_info=True)
            return None

    def get_task(self, task_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve a task by ID

        Args:
            task_id: Task identifier

        Returns:
            Task data dictionary or None if not found
        """
        try:
            logger.debug(f"Querying task {task_id} from Supabase")
            result = self.client.table("tasks").select("*").eq("id", str(task_id)).maybe_single().execute()

            logger.debug(f"Query result for task {task_id}: data={result.data}")

            if result.data:
                logger.info(f"Task {task_id} retrieved successfully")
                # Ensure datetime fields are present even if None
                task_data = result.data
                if 'created_at' not in task_data:
                    task_data['created_at'] = None
                if 'updated_at' not in task_data:
                    task_data['updated_at'] = None
                if 'completed_at' not in task_data:
                    task_data['completed_at'] = None
                return task_data
            logger.warning(f"Task {task_id} not found in database")
            return None
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}", exc_info=True)
            return None

    def update_task_status(
        self,
        task_id: UUID,
        status: TaskStatus,
        result_video_url: Optional[str] = None,
        error_message: Optional[str] = None,
        file_size: Optional[int] = None
    ) -> bool:
        """
        Update task status and related fields

        Args:
            task_id: Task identifier
            status: New status
            result_video_url: Public URL of processed video
            error_message: Error message if failed
            file_size: Total file size processed

        Returns:
            True if updated successfully
        """
        try:
            update_data: Dict[str, Any] = {
                "status": status.value,
                "updated_at": datetime.utcnow().isoformat()  # Always update this timestamp
            }

            if result_video_url:
                update_data["result_video_url"] = result_video_url

            if error_message:
                update_data["error_message"] = error_message

            if file_size:
                update_data["file_size"] = file_size

            if status in [TaskStatus.SUCCESS, TaskStatus.FAILED]:
                update_data["completed_at"] = datetime.utcnow().isoformat()

            self.client.table("tasks").update(update_data).eq("id", str(task_id)).execute()

            logger.info(f"Updated task {task_id} status to {status.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            return False

    def get_old_tasks(self, hours: int = 2) -> List[Dict[str, Any]]:
        """
        Get tasks older than specified hours for cleanup

        Args:
            hours: Age threshold in hours

        Returns:
            List of task data dictionaries
        """
        try:
            result = self.client.rpc(
                "get_old_tasks",
                {"hours_old": hours}
            ).execute()

            if result.data:
                return result.data
            return []
        except Exception as e:
            logger.warning(f"Failed to get old tasks (may not have RPC function): {e}")
            # Fallback: just query completed tasks
            try:
                result = self.client.table("tasks").select("*").in_(
                    "status", ["success", "failed"]
                ).execute()
                return result.data if result.data else []
            except Exception as e2:
                logger.error(f"Fallback query also failed: {e2}")
                return []

    def delete_task(self, task_id: UUID) -> bool:
        """
        Delete a task record

        Args:
            task_id: Task identifier

        Returns:
            True if deleted successfully
        """
        try:
            self.client.table("tasks").delete().eq("id", str(task_id)).execute()
            logger.info(f"Deleted task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            return False


supabase_service = SupabaseService()
