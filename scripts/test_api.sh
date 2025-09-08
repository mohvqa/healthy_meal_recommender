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

echo -e "\n=== GET /top-bias?k=${K} ==="
http GET "${BASE}/top-bias" k==${K}

echo -e "\n=== GET /similar/1?k=${K} ==="
http GET "${BASE}/similar/${MEALID}" k==${K}