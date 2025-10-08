import json
import re
import requests
from typing import Any, Dict, Optional
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

    def _openai_base(self) -> str:
        # Normalize to OpenAI-compatible base path
        # Accept both https://api.groq.com/v1 and https://api.groq.com/openai/v1
        if self.api_endpoint.rstrip('/').endswith('/openai/v1'):
            return self.api_endpoint.rstrip('/')
        if self.api_endpoint.rstrip('/').endswith('/v1'):
            # Prefer openai-compatible route
            return self.api_endpoint.rstrip('/').replace('/v1', '/openai/v1')
        # Fallback
        return f"{self.api_endpoint.rstrip('/')}/openai/v1"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def send_inference_request(self, model_id: str, input_data: Dict[str, Any]) -> InferenceResponse:
        # OpenAI-compatible chat completions API
        payload = {
            "model": model_id,
            "messages": input_data.get("messages", [])
        }
        response = requests.post(
            f"{self._openai_base()}/chat/completions",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        # Map generic response to our InferenceResponse model
        try:
            prediction = {
                "message": data.get("choices", [{}])[0].get("message", {}).get("content", "")
            }
            return InferenceResponse.parse_obj({
                "prediction": prediction,
                "confidence": 0.75,
                "status": "success"
            })
        except ValidationError as e:
            print("Validation Error:", e)
            raise

    def send_code_review_request(self, model_id: str, code_review_request: CodeReviewRequest) -> CodeReviewFeedback:
        # Use chat completions to perform a structured code review and parse JSON
        system_prompt = (
            "You are a strict code review assistant. Return ONLY a JSON object with keys: "
            "issues (array of {description: string, severity: string}), suggestions (array of string), "
            "overall_quality (string). Do not include any additional text."
        )
        user_prompt = (
            f"Review the following file. Name: {code_review_request.file_name}.\n"
            f"Diff (if any):\n{code_review_request.diff}\n\n"
            f"Full content:\n{code_review_request.file_content}"
        )
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        response = requests.post(
            f"{self._openai_base()}/chat/completions",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

        def _extract_json(text: str) -> Optional[Dict[str, Any]]:
            # Try direct parse
            try:
                return json.loads(text)
            except Exception:
                pass
            # Extract fenced JSON
            match = re.search(r"```(?:json)?\n([\s\S]*?)\n```", text)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    return None
            return None

        parsed = _extract_json(content) or {
            "issues": [],
            "suggestions": [],
            "overall_quality": "Unknown"
        }
        try:
            return CodeReviewFeedback.parse_obj(parsed)
        except ValidationError:
            # Coerce to expected structure if partial
            sanitized = {
                "issues": parsed.get("issues", []),
                "suggestions": parsed.get("suggestions", []),
                "overall_quality": parsed.get("overall_quality", "Unknown")
            }
            return CodeReviewFeedback.parse_obj(sanitized)

    # New Method for Chat-Create API (OpenAI-compatible under the hood)
    def send_chat_create_request(self, chat_create_request: ChatCreateRequest, model_id: str = "llama3-8b-8192") -> ChatCreateResponse:
        messages = []
        if chat_create_request.context:
            messages.append({"role": "system", "content": json.dumps(chat_create_request.context)})
        messages.append({"role": "user", "content": chat_create_request.user_message})
        payload = {
            "model": model_id,
            "messages": messages
        }
        response = requests.post(
            f"{self._openai_base()}/chat/completions",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            return ChatCreateResponse.parse_obj({
                "bot_response": content,
                "confidence": 0.75,
                "status": "success"
            })
        except ValidationError as e:
            print("Validation Error:", e)
            raise

    # Optional config query used by some agents; return None to skip remote config.
    def query(self, _query: str) -> Optional[Dict[str, Any]]:
        return None