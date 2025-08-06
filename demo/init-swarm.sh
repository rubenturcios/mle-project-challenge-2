function init() {
    docker swarm init > /dev/null 2>&1 || echo "Already in swarm mode"
    docker service create --name registry --publish published=5000,target=5000 registry:2 > /dev/null 2>&1 || echo "Registry service already exists."
    docker network create --driver overlay --attachable proxy > /dev/null 2>&1 || echo "Proxy network already exists."
    docker compose build && docker compose push > /dev/null 2>&1
}

function main() {

    docker stack deploy -c swarm/model-compose.yml model
    docker stack deploy -c swarm/app-compose.yml app

    sleep 2

    echo "\nChecking which model container we are using..."
    echo "$(curl -s http://localhost:8008/check-model-host)"

    echo "\nScaling model service to 3..."
    docker service scale model_model=3

    echo "\nChecking which model containers we are using..."
    echo "$(curl -s http://localhost:8008/check-model-host)"
    echo "$(curl -s http://localhost:8008/check-model-host)"
    echo "$(curl -s http://localhost:8008/check-model-host)"

    echo "\nSpinning up new model service..."
    docker service create --network proxy --name model_2 --env MODEL_FILE=model/model.pkl --mount 'type=bind,source=./model,target=/code/model' 127.0.0.1:5000/model:latest fastapi run app/model.py --port 80

    echo "\nCreating new nginx conf file..."
    sed -e 's/model:/model_2:/g' ./nginx/nginx.conf > ./nginx/temp.conf
    docker config rm new_conf_nginx > /dev/null 2>&1 || echo "New config does not exist, creating."

    docker config create new_conf_nginx ./nginx/temp.conf > /dev/null 2>&1

    echo "\n"
    docker ps
    echo "\n"

    echo "\nUpdating nginx service to use new conf file..."
    docker service update --config-rm nginx-config --config-add source=new_conf_nginx,target=/etc/nginx/conf.d/default.conf app_nginx

    echo "\nChecking which model container we are using..."
    echo "$(curl -s http://localhost:8008/check-model-host)"
}

function destroy() {
    docker stack rm model > /dev/null 2>&1
    docker stack rm app > /dev/null 2>&1
    docker service rm model_2 > /dev/null 2>&1

    docker config rm new_conf_nginx > /dev/null 2>&1

    rm ./nginx/temp.conf > /dev/null 2>&1

    Echo "Autoscaling demo complete."
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
