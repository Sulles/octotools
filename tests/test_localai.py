import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import uvicorn
import threading
import time
from octotools.engine.localai import ChatLocalAI
import json
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Create FastAPI app that mimics OpenAI's endpoints
app = FastAPI()

# Add AnswerVerification model definition
class AnswerVerification(BaseModel):
    analysis: str
    true_false: bool

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """Mock OpenAI chat completions endpoint"""
    print("\n--- Regular Chat Completions ---")
    print(f"Received request: {json.dumps(request, indent=2)}")
    
    # Check if this is a structured request
    if request.get("response_format") is not None:
        print(f"Got structured request with schema: {request.get('response_format')}")
        
        # Create response matching AnswerVerification format
        structured_response = {
            "analysis": "This is a test analysis",
            "true_false": True
        }
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": json.dumps(structured_response),  # JSON string in content
                    "function_call": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "model": request.get("model"),
            "object": "chat.completion",
        }
    else:
        return {
            "choices": [{
                "message": {
                    "content": "This is a test response",
                    "role": "assistant"
                },
                "finish_reason": "stop",
                "index": 0
            }]
        }

# Test server setup and teardown
@pytest.fixture(scope="module")
def test_server():
    # Start server in a separate thread
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8080, log_level="error")
    )
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    time.sleep(0.1)
    
    # Create test client
    client = TestClient(app)
    yield client
    
    # Cleanup not needed as thread is daemonic

# @pytest.mark.skip(reason="Test works, skipping for now")
def test_basic_generation(test_server: TestClient):
    """Test basic text generation"""
    # Verify server is responding
    print("\n=== Starting Basic Generation Test ===")
    response = test_server.get("/")
    assert response.status_code == 404  # FastAPI default for undefined root
    
    llm = ChatLocalAI(
        base_url="http://localhost:8080/v1",
        model_string="test-model",
        enable_cache=False
    )
    
    response = llm.generate("Hello, world!")
    print(f"\nReceived response: {type(response)} \n{response}")
    assert response == "This is a test response"

def test_structured_generation(test_server: TestClient):
    """Test structured output generation"""
    print("\n=== Starting Structured Generation Test ===")

    # Verify server is responding
    response = test_server.get("/")
    assert response.status_code == 404  # FastAPI default for undefined root
    
    llm = ChatLocalAI(
        base_url="http://localhost:8080/v1",
        model_string="gpt-4o-mini",  # Use a structured model
        enable_cache=False
    )
    
    print("\nMaking request with parameters:")
    print(f"Model: {llm.model_string}")
    print(f"Base URL: {llm.client.base_url}")
    
    response = llm.generate(
        "Hello, world!",
        response_format=AnswerVerification
    )
    
    print(f"\nReceived response: {response}")
    
    assert isinstance(response, AnswerVerification)
    assert response.analysis == "This is a test analysis"
    assert response.true_false is True

def test_multimodal_generation(test_server: TestClient):
    """Test multimodal generation with image"""
    print("\n=== Starting Multimodal Generation Test ===")

    # Verify server is responding
    response = test_server.get("/")
    assert response.status_code == 404  # FastAPI default for undefined root

    llm = ChatLocalAI(
        base_url="http://localhost:8080/v1",
        model_string="test-model",
        is_multimodal=True,
        enable_cache=False
    )
    
    # Create dummy image bytes
    image_bytes = b"dummy image data"
    response = llm.generate([
        "Describe this image:",
        image_bytes
    ])
    assert response == "This is a test response"

def test_error_handling(test_server: TestClient):
    """Test error handling"""
    print("\n=== Starting Error Handling Test ===")

    # Verify server is responding
    response = test_server.get("/")
    assert response.status_code == 404  # FastAPI default for undefined root

    llm = ChatLocalAI(
        base_url="http://localhost:8080/v1",
        model_string="test-model",
        enable_cache=False
    )
    
    # Test with invalid URL to trigger connection error
    llm.client.base_url = "http://invalid-url:8080/v1"
    response = llm.generate("Hello, world!")
    assert "error" in response
    assert response["error"] == "APIConnectionError"
