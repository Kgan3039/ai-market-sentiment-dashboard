#!/bin/bash
# API Testing Script using curl
# Run: chmod +x test_api_curl.sh && ./test_api_curl.sh

BASE_URL="http://localhost:8000"
echo "🚀 Testing AI Market Sentiment API with curl"
echo "==========================================="

# Function to test endpoint
test_endpoint() {
    local name="$1"
    local method="$2"
    local url="$3"
    local expected_status="${4:-200}"

    echo ""
    echo "🧪 Testing $name..."
    echo "   $method $url"

    if [ "$method" = "POST" ]; then
        response=$(curl -s -w "\nHTTPSTATUS:%{http_code}" -X POST "$url" \
                  -H "Content-Type: application/json" \
                  -d "{\"text\": \"$5\"}" 2>/dev/null)
    else
        response=$(curl -s -w "\nHTTPSTATUS:%{http_code}" "$url" 2>/dev/null)
    fi

    http_code=$(echo "$response" | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    body=$(echo "$response" | sed -e 's/HTTPSTATUS:.*//g')

    if [ "$http_code" = "$expected_status" ]; then
        echo "   ✅ Status: $http_code"
        echo "   📄 Response: ${body:0:100}..."
    else
        echo "   ❌ Status: $http_code (expected $expected_status)"
        echo "   📄 Response: ${body:0:100}..."
    fi
}

# Wait for server
echo "⏳ Waiting for server..."
sleep 2

# Test endpoints
test_endpoint "Health Check" "GET" "$BASE_URL/test"
test_endpoint "Sentiment NVDA" "GET" "$BASE_URL/sentiment/NVDA"
test_endpoint "Sentiment TSLA" "GET" "$BASE_URL/sentiment/TSLA"
test_endpoint "Text Analysis" "POST" "$BASE_URL/sentiment/analyze-text" 200 "NVDA stock is performing well"
test_endpoint "Prediction NVDA" "GET" "$BASE_URL/prediction/NVDA"
test_endpoint "Market Data AAPL" "GET" "$BASE_URL/market/AAPL"
test_endpoint "Dashboard NVDA" "GET" "$BASE_URL/dashboard/summary/NVDA"

echo ""
echo "🔗 Useful URLs:"
echo "   📚 Interactive Docs: $BASE_URL/docs"
echo "   📖 ReDoc: $BASE_URL/redoc"
echo "   🏥 Health Check: $BASE_URL/test"