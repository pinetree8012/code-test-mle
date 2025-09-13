DATA='{"amount": 120.0, "time": 13, "mismatch": 0, "frequency": 3}'

call_ms() {
  curl -s -o /dev/null -w "%{time_total}\n" \
    -H "Content-Type: application/json" \
    -d "$DATA" "http://localhost:8000/predict"
}

echo "Cold start"
call_ms

echo "Warm-up: 10 consecutive calls"
for i in {1..10}; do call_ms; done
