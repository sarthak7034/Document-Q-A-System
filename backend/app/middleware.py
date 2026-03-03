"""
Middleware for error handling and request logging.
"""

import logging
import time
import traceback
from typing import Callable
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.
    
    Catches unhandled exceptions and returns appropriate JSON error responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and handle any unhandled exceptions.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            Response object
        """
        try:
            response = await call_next(request)
            return response
            
        except Exception as exc:
            # Log the full exception with stack trace
            logger.error(
                f"Unhandled exception in {request.method} {request.url.path}: {exc}",
                exc_info=True
            )
            
            # Return JSON error response
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "details": {
                        "type": type(exc).__name__,
                        "message": str(exc)
                    }
                }
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware.
    
    Logs all API requests with timestamp, endpoint, method, status code,
    and response time.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log details.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            Response object
        """
        # Record start time
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            f"Incoming request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- Status: {response.status_code} "
                f"- Time: {process_time:.3f}s"
            )
            
            # Add response time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as exc:
            # Calculate response time even for errors
            process_time = time.time() - start_time
            
            # Log error with stack trace
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- Error: {type(exc).__name__}: {exc} "
                f"- Time: {process_time:.3f}s",
                exc_info=True
            )
            
            # Re-raise to be handled by error handling middleware
            raise
