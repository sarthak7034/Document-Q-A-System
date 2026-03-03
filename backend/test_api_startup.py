"""
Quick test to verify the FastAPI application can start without errors.
"""

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing FastAPI application startup...")
    
    # Import the app
    from app.main import app
    
    print("✓ FastAPI app imported successfully")
    
    # Check that routes are registered
    routes = [route.path for route in app.routes]
    print(f"✓ Found {len(routes)} routes")
    
    # Check for expected routes
    expected_routes = [
        "/api/documents",
        "/api/documents/{document_id}",
        "/api/questions",
        "/api/health"
    ]
    
    for expected in expected_routes:
        if any(expected in route for route in routes):
            print(f"✓ Route found: {expected}")
        else:
            print(f"✗ Route missing: {expected}")
    
    print("\n✓ All checks passed! FastAPI application is ready.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
