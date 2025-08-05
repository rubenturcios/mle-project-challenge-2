function init() {
    docker compose build > /dev/null 2>&1
    docker compose up -d --remove-orphans
    sleep 2
}

function main() {
    echo "\nChecking which model container we are using..."
    echo "$(curl -s http://localhost:8008/check-model-host)"

    echo "Spinning up a new model container..."
    docker run --network mle-project-challenge-2_default -e MODEL_FILE=model/model.pkl --name model-2 -d 127.0.0.1:5000/model fastapi run app/model.py --port 80

    echo "\nSwitching model container..."
    ./demo/reload.py -m model-2

    sleep 2

    echo "\nChecking which model container we are using..."
    echo "$(curl -s http://localhost:8008/check-model-host)"
}

function destroy() {
    echo "Exiting..."

    docker stop model-2 > /dev/null 2>&1
    docker rm model-2 > /dev/null 2>&1
    docker compose down

    echo "\n\"Blue-Green\" deployment demo complete."
}

if [[ -z "$1" ]]; then
    init
    main
    destroy
elif [[ $1 == 'keep' ]]; then
    init
    main
elif [[ $1 == 'destroy' ]]; then
    destroy
fi
