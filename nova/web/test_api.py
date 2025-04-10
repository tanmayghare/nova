"""API testing utilities for Nova dashboard."""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

logger = logging.getLogger(__name__)

class NovaApiTest:
    """Test client for Nova API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize the API test client.
        
        Args:
            base_url: Base URL for the API
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=10.0)
    
    async def test_info_endpoint(self) -> Tuple[bool, Dict[str, Any]]:
        """Test the /api/info endpoint.
        
        Returns:
            Tuple of (success, result)
        """
        try:
            response = await self.client.get("/api/info")
            result = {"status_code": response.status_code, "content": response.json()}
            return response.status_code == 200, result
        except Exception as e:
            logger.error(f"Error testing info endpoint: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    async def test_tasks_listing(self) -> Tuple[bool, Dict[str, Any]]:
        """Test the GET /api/tasks endpoint.
        
        Returns:
            Tuple of (success, result)
        """
        try:
            response = await self.client.get("/api/tasks")
            result = {"status_code": response.status_code, "content": response.json()}
            return response.status_code == 200, result
        except Exception as e:
            logger.error(f"Error testing tasks listing: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    async def test_task_creation(self, task: str = "Test task") -> Tuple[bool, Dict[str, Any]]:
        """Test the POST /api/tasks endpoint.
        
        Args:
            task: Task description
            
        Returns:
            Tuple of (success, result)
        """
        try:
            data = {
                "task": task,
                "model": "llama3.2:3b-instruct-q8_0",
                "headless": True
            }
            response = await self.client.post("/api/tasks", json=data)
            result = {"status_code": response.status_code, "content": response.json()}
            return response.status_code == 201, result
        except Exception as e:
            logger.error(f"Error testing task creation: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    async def test_task_retrieval(self, task_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Test the GET /api/tasks/{task_id} endpoint.
        
        Args:
            task_id: Task ID to retrieve
            
        Returns:
            Tuple of (success, result)
        """
        try:
            response = await self.client.get(f"/api/tasks/{task_id}")
            result = {"status_code": response.status_code, "content": response.json()}
            return response.status_code == 200, result
        except Exception as e:
            logger.error(f"Error testing task retrieval: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    async def test_stats_endpoint(self) -> Tuple[bool, Dict[str, Any]]:
        """Test the /monitoring/stats endpoint.
        
        Returns:
            Tuple of (success, result)
        """
        try:
            response = await self.client.get("/monitoring/stats")
            result = {"status_code": response.status_code, "content": response.json()}
            return response.status_code == 200, result
        except Exception as e:
            logger.error(f"Error testing stats endpoint: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    async def test_history_endpoint(self) -> Tuple[bool, Dict[str, Any]]:
        """Test the /monitoring/history endpoint.
        
        Returns:
            Tuple of (success, result)
        """
        try:
            response = await self.client.get("/monitoring/history")
            result = {"status_code": response.status_code, "content": response.json()}
            return response.status_code == 200, result
        except Exception as e:
            logger.error(f"Error testing history endpoint: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    async def test_pages(self) -> Dict[str, Tuple[bool, Dict[str, Any]]]:
        """Test all HTML pages.
        
        Returns:
            Dict of page results
        """
        results = {}
        pages = ["/", "/tasks", "/builder", "/settings"]
        
        for page in pages:
            try:
                response = await self.client.get(page)
                success = response.status_code == 200
                result = {
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "length": len(response.content)
                }
                results[page] = (success, result)
            except Exception as e:
                logger.error(f"Error testing page {page}: {e}", exc_info=True)
                results[page] = (False, {"error": str(e)})
        
        return results
    
    async def test_websocket(self, timeout: float = 5.0) -> Tuple[bool, Dict[str, Any]]:
        """Test WebSocket connection.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, result)
        """
        try:
            ws_url = f"{self.base_url.replace('http', 'ws')}/monitoring/ws"
            ws_client = httpx.AsyncClient()
            
            async with ws_client:
                # This is a simple check if the endpoint exists and responds correctly.
                # For actual WebSocket testing, we would need a more comprehensive approach.
                response = await ws_client.get(ws_url)
                if response.status_code == 101:  # Switching Protocols (WebSocket handshake)
                    return True, {"status": "WebSocket endpoint available"}
                return False, {"status_code": response.status_code, "content": response.text}
        except Exception as e:
            logger.error(f"Error testing WebSocket: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests.
        
        Returns:
            Dict of test results
        """
        results = {}
        
        # Test API endpoints
        results["info_endpoint"] = await self.test_info_endpoint()
        results["tasks_listing"] = await self.test_tasks_listing()
        results["stats_endpoint"] = await self.test_stats_endpoint()
        results["history_endpoint"] = await self.test_history_endpoint()
        
        # Test pages
        results["pages"] = await self.test_pages()
        
        # Test task creation
        task_success, task_result = await self.test_task_creation()
        results["task_creation"] = (task_success, task_result)
        
        if task_success:
            # Test task retrieval if creation was successful
            task_id = task_result["content"]["task_id"]
            results["task_retrieval"] = await self.test_task_retrieval(task_id)
        
        # Test WebSocket
        results["websocket"] = await self.test_websocket()
        
        return results
    
    async def close(self) -> None:
        """Close the client."""
        await self.client.aclose()


async def run_tests(base_url: str = "http://localhost:8000") -> None:
    """Run API tests.
    
    Args:
        base_url: Base URL for the API
    """
    tester = NovaApiTest(base_url)
    try:
        results = await tester.run_all_tests()
        
        print("\n=== Nova API Testing Results ===\n")
        
        for test_name, (success, result) in results.items():
            if test_name != "pages":
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"{status} - {test_name}: {json.dumps(result, indent=2)}")
            else:
                print("\n--- Page Tests ---")
                for page, (page_success, page_result) in result.items():
                    page_status = "✅ PASS" if page_success else "❌ FAIL"
                    print(f"{page_status} - {page}: {json.dumps(page_result, indent=2)}")
        
        print("\n=== End of Testing Results ===\n")
    finally:
        await tester.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Nova API endpoints")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for the Nova API")
    
    args = parser.parse_args()
    
    asyncio.run(run_tests(args.url)) 