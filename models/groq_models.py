from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class GitHubAction(BaseModel):
    name: str
    steps: List[str]

class DockerConfig(BaseModel):
    base_image: str
    expose_port: int
    copy_source: str
    work_dir: str

class InferenceRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Any]

class InferenceResponse(BaseModel):
    prediction: Dict[str, Any]
    confidence: float
    status: str

# New Models for Code Review
class CodeReviewRequest(BaseModel):
    file_name: str
    file_content: str
    diff: str  # The diff of the changes

class CodeReviewFeedback(BaseModel):
    issues: List[Dict[str, Any]]
    suggestions: List[str]
    overall_quality: str

# New Models for Chat-Create API
class ChatCreateRequest(BaseModel):
    user_message: str
    context: Optional[Dict[str, Any]] = None  # Additional context if needed

class ChatCreateResponse(BaseModel):
    bot_response: str
    confidence: float
    status: str