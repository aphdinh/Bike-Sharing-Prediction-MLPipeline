#!/bin/bash
BASE="http://localhost:8000"
NORMAL='{"date":"15/06/2018","hour":8,"temperature_c":20.0,"humidity":60.0,"wind_speed":2.0,"visibility_10m":1500.0,"dew_point_c":10.0,"solar_radiation":1.5,"rainfall_mm":0.0,"snowfall_cm":0.0,"season":"Summer","holiday":"No Holiday","functioning_day":"Yes"}'
EXTREME='{"date":"15/07/2018","hour":14,"temperature_c":40.0,"humidity":99.0,"wind_speed":18.0,"visibility_10m":50.0,"dew_point_c":35.0,"solar_radiation":0.1,"rainfall_mm":80.0,"snowfall_cm":0.0,"season":"Summer","holiday":"No Holiday","functioning_day":"Yes"}'

echo "--- normal prediction ---"
curl -s -X POST $BASE/predict -H 'Content-Type: application/json' -d "$NORMAL" | python3 -m json.tool

echo "--- injecting 10 extreme predictions into buffer ---"
for i in $(seq 1 10); do
  curl -s -X POST $BASE/predict -H 'Content-Type: application/json' -d "$EXTREME" > /dev/null
done
echo "done"

echo "--- drift check ---"
curl -s -X POST $BASE/monitoring/data-drift | python3 -m json.tool
