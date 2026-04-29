source ~/cred.sh
python -m webwright.run.mini \
    -c mini.yaml -c model_openai.yaml \
    -t "Find the cheapest economy flight from SEA to JFK on 2026-05-15" \
    --start-url https://www.google.com/flights \
    --task-id demo_openai \
    -o outputs/default