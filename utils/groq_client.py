import requests
from typing import Any, Dict
from pydantic import ValidationError
from models.groq_models import (
    InferenceRequest,
    InferenceResponse,
    CodeReviewRequest,
    CodeReviewFeedback,
    ChatCreateRequest,
    ChatCreateResponse
)

class GROQClient:
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key

    def send_inference_request(self, model_id: str, input_data: Dict[str, Any]) -> InferenceResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_id,
            "messages": input_data["messages"]
        }
        response = requests.post(self.api_endpoint, json=payload, headers=headers)
        response.raise_for_status()
        try:
            return InferenceResponse.parse_obj(response.json())
        except ValidationError as e:
            print("Validation Error:", e)
            raise

    def send_code_review_request(self, model_id: str, code_review_request: CodeReviewRequest) -> CodeReviewFeedback:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model_id": model_id,
            "input_data": code_review_request.dict()
        }
        response = requests.post(f"{self.api_endpoint}/code-review", json=payload, headers=headers)
        response.raise_for_status()
        try:
            return CodeReviewFeedback.parse_obj(response.json())
        except ValidationError as e:
            print("Validation Error:", e)
            raise

    # New Method for Chat-Create API
    def send_chat_create_request(self, chat_create_request: ChatCreateRequest) -> ChatCreateResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "user_message": chat_create_request.user_message,
            "context": chat_create_request.context
        }
        response = requests.post(self.api_endpoint, json=payload, headers=headers)
        response.raise_for_status()
        try:
            return ChatCreateResponse.parse_obj(response.json())
        except ValidationError as e:
            print("Validation Error:", e)
            raise