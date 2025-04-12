"""Tests for NVIDIA NIM provider integration."""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, patch
from nova.core.nim_provider import NIMProvider

@pytest.fixture
def mock_env():
    """Mock environment variables."""
    with patch.dict(os.environ, {"NVIDIA_NIM_API_KEY": "test-key"}):
        yield

@pytest.mark.asyncio
async def test_nim_provider_initialization(mock_env):
    """Test that the NIM provider initializes correctly."""
    provider = NIMProvider()
    assert provider is not None
    assert provider.config.model_name == "mistral-7b"
    assert provider.config.temperature == 0.2
    assert provider.config.max_tokens == 4096

@pytest.mark.asyncio
async def test_nim_provider_generate(mock_env):
    """Test text generation with NIM provider."""
    provider = NIMProvider()
    
    # Mock the API response
    mock_response = {
        "choices": [{
            "message": {
                "content": "Test response"
            }
        }]
    }
    
    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        
        response = await provider.generate("Test prompt")
        assert response == "Test response"

@pytest.mark.asyncio
async def test_nim_provider_generate_plan(mock_env):
    """Test plan generation with NIM provider."""
    provider = NIMProvider()
    
    # Mock the API response with a valid plan
    mock_response = {
        "choices": [{
            "message": {
                "content": """{
                    "steps": [
                        {"tool": "navigate", "input": {"url": "https://example.com"}},
                        {"tool": "click", "input": {"selector": "#submit"}}
                    ]
                }"""
            }
        }]
    }
    
    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        
        plan = await provider.generate_plan("Test task", "Test context")
        assert len(plan) == 2
        assert plan[0]["tool"] == "navigate"
        assert plan[1]["tool"] == "click"

@pytest.mark.asyncio
async def test_nim_provider_generate_stream(mock_env):
    """Test streaming generation with NIM provider."""
    provider = NIMProvider()
    
    # Mock the streaming response
    mock_chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": " World"}}]}\n\n'
    ]
    
    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks
            self.status = 200
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, *args):
            pass
            
        @property
        def content(self):
            async def iter_chunks():
                for chunk in self.chunks:
                    yield chunk
            return iter_chunks()
    
    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.__aenter__.return_value = MockResponse(mock_chunks)
        
        chunks = []
        async for chunk in provider.generate_stream("Test prompt"):
            chunks.append(chunk)
            
        assert "".join(chunks) == "Hello World"

@pytest.mark.asyncio
async def test_nim_provider_error_handling(mock_env):
    """Test error handling in NIM provider."""
    provider = NIMProvider()
    
    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 500
        mock_post.return_value.__aenter__.return_value.text = AsyncMock(return_value="Server error")
        
        with pytest.raises(RuntimeError):
            await provider.generate("Test prompt")

@pytest.mark.asyncio
async def test_nim_provider_circuit_breaker(mock_env):
    """Test circuit breaker functionality."""
    provider = NIMProvider()
    
    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 500
        mock_post.return_value.__aenter__.return_value.text = AsyncMock(return_value="Server error")
        
        # First 5 failures should be retried
        for _ in range(5):
            with pytest.raises(RuntimeError):
                await provider.generate("Test prompt")
        
        # After 5 failures, circuit breaker should open
        with pytest.raises(RuntimeError, match="Circuit breaker opened"):
            await provider.generate("Test prompt") 