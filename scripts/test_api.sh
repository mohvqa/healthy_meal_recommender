source venv/bin/activate
# export vars from .env
set -a; source .env; set +a

BASE="http://$HOST:$PORT"

# recommend to user id
USERID=42

# get similar meals to meal id
MEALID=1

K=2

echo "=== POST /recommend ==="
http POST "${BASE}/recommend" user_id:=${USERID} k:=${K}

echo -e "\n=== POST /recommend-cold ==="
http POST "${BASE}/recommend-cold" \
  Gender="Female" \
  "Activity Level"="Moderately Active" \
  Ages:=30 \
  Height:=165.0 \
  Weight:=65.0 \
  "Daily Calorie Target":=2000 \
  Protein:=100.0 \
  Sugar:=50.0 \
  Sodium:=2000.0 \
  Calories:=2000.0 \
  Carbohydrates:=250.0 \
  Fiber:=30.0 \
  Fat:=70.0 \
  Acne:=0 \
  Diabetes:=0 \
  "Heart Disease":=0 \
  Hypertension:=0 \
  "Kidney Disease":=0 \
  "Weight Gain":=0 \
  "Weight Loss":=1 \
  k:=${K}

echo -e "\n=== GET /top-bias?k=${K} ==="
http GET "${BASE}/top-bias" k==${K}

echo -e "\n=== GET /similar/1?k=${K} ==="
http GET "${BASE}/similar/${MEALID}" k==${K}