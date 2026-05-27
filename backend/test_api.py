#!/usr/bin/env python3
"""
API Testing Script for AI Market Sentiment Dashboard Backend

This script provides comprehensive testing of all endpoints without relying
on other team members' code. All tests use mock/placeholder data.

Usage:
    python test_api.py

Requirements:
    pip install requests
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 10

def test_endpoint(name: str, method: str, url: str, expected_status: int = 200, **kwargs) -> bool:
    """Test a single endpoint and return success status."""
    print(f"\n🧪 Testing {name}...")
    print(f"   {method} {url}")

    try:
        response = requests.request(method, url, timeout=TIMEOUT, **kwargs)

        if response.status_code == expected_status:
            print(f"   ✅ Status: {response.status_code}")
            try:
                data = response.json()
                print(f"   📄 Response: {json.dumps(data, indent=2)[:200]}...")
                return True
            except json.JSONDecodeError:
                print(f"   📄 Response: {response.text[:200]}...")
                return True
        else:
            print(f"   ❌ Status: {response.status_code} (expected {expected_status})")
            print(f"   📄 Response: {response.text[:200]}...")
            return False

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error: {e}")
        return False


test_endpoint.__test__ = False

def main():
    """Run all API tests."""
    print("🚀 AI Market Sentiment API Testing Suite")
    print("=" * 50)

    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)

    tests_passed = 0
    total_tests = 0

    # Test 1: Health Check
    total_tests += 1
    if test_endpoint("Health Check", "GET", f"{BASE_URL}/test"):
        tests_passed += 1

    # Test 2: Sentiment Analysis
    total_tests += 1
    if test_endpoint("Sentiment - NVDA", "GET", f"{BASE_URL}/sentiment/NVDA"):
        tests_passed += 1

    total_tests += 1
    if test_endpoint("Sentiment - TSLA", "GET", f"{BASE_URL}/sentiment/TSLA"):
        tests_passed += 1

    # Test 3: Text Sentiment Analysis
    total_tests += 1
    test_text = "NVDA earnings beat expectations by 15%"
    if test_endpoint(
        "Text Sentiment Analysis",
        "POST",
        f"{BASE_URL}/sentiment/analyze-text",
        json={"text": test_text}
    ):
        tests_passed += 1

    # Test 4: Stock Predictions
    total_tests += 1
    if test_endpoint("Prediction - NVDA", "GET", f"{BASE_URL}/prediction/NVDA"):
        tests_passed += 1

    total_tests += 1
    if test_endpoint("Prediction - TSLA", "GET", f"{BASE_URL}/prediction/TSLA"):
        tests_passed += 1

    # Test 5: Market Data
    total_tests += 1
    if test_endpoint("Market Data - AAPL", "GET", f"{BASE_URL}/market/AAPL"):
        tests_passed += 1

    total_tests += 1
    if test_endpoint("Market Data - GOOGL", "GET", f"{BASE_URL}/market/GOOGL"):
        tests_passed += 1

    # Test 6: Batch Market Data
    total_tests += 1
    if test_endpoint(
        "Batch Market Data",
        "GET",
        f"{BASE_URL}/market/batch",
        params={"tickers": ["NVDA", "TSLA", "AAPL"]}
    ):
        tests_passed += 1

    # Test 7: Dashboard Summary
    total_tests += 1
    if test_endpoint("Dashboard Summary - NVDA", "GET", f"{BASE_URL}/dashboard/summary/NVDA"):
        tests_passed += 1

    total_tests += 1
    if test_endpoint("Dashboard Summary - TSLA", "GET", f"{BASE_URL}/dashboard/summary/TSLA"):
        tests_passed += 1

    # Test 8: Batch Dashboard Summary
    total_tests += 1
    if test_endpoint(
        "Batch Dashboard Summary",
        "GET",
        f"{BASE_URL}/dashboard/summary-batch",
        params=[("tickers", "NVDA"), ("tickers", "TSLA")]
    ):
        tests_passed += 1

    # Test 9: Error Handling
    total_tests += 1
    if test_endpoint("Invalid Ticker", "GET", f"{BASE_URL}/sentiment/", expected_status=404):
        tests_passed += 1

    total_tests += 1
    if test_endpoint("Empty Text", "POST", f"{BASE_URL}/sentiment/analyze-text", json={"text": ""}, expected_status=422):
        tests_passed += 1

    # Results
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   ✅ Passed: {tests_passed}/{total_tests}")
    print(f"   ❌ Failed: {total_tests - tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Your API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    # Additional info
    print("\n🔗 Useful URLs:")
    print(f"   📚 Interactive Docs: {BASE_URL}/docs")
    print(f"   📖 ReDoc: {BASE_URL}/redoc")
    print(f"   🏥 Health Check: {BASE_URL}/test")

if __name__ == "__main__":
    main()
