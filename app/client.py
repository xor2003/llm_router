import httpx
from typing import Any, Dict
from app.config import ModelDeployment  # Updated import


class LLMClient:
    async def make_request(
        self, deployment: ModelDeployment, payload: Dict[str, Any]
    ) -> httpx.Response:
        """Send request to the LLM deployment endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                deployment.endpoint_url,  # Now using attribute from ModelDeployment
                json=payload,
                headers=deployment.headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response

    def adapt_gemini_response(
        self, response_json: Dict[str, Any], model_name: str
    ) -> Dict[str, Any]:
        """Adapt Gemini responses to OpenAI format"""
        # Basic adaptation - needs improvement based on actual Gemini response format
        if "candidates" in response_json:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": candidate["content"]["parts"][0]["text"],
                        }
                    }
                    for candidate in response_json["candidates"]
                ]
            }
        return response_json
