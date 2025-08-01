#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
from typing import Protocol
import os
import subprocess


NGINX_CONF_PATH = os.getenv("NGINX_CONF_PATH", "nginx/nginx.conf")

conf_file = """upstream loadbalancer-app {{
  server {app_hostname}:{port};
}}

upstream loadbalancer-model {{
  server {model_hostname}:{port};
}}

server {{
  listen 80;
  server_name localhost;
  location / {{
    proxy_pass http://loadbalancer-app;
  }}
}}

server {{
  listen 81;
  server_name localhost;
  location / {{
    proxy_pass http://loadbalancer-model;
  }}
}}"""


class Args(Protocol):
    model_hostname: str
    app_hostname: str
    nginx_name: str
    env_file_path: str
    port: str


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--model-hostname", "-m",
        action="store",
        default="model"
    )
    parser.add_argument(
        "--app-hostname", "-a",
        action="store",
        default="app"
    )
    parser.add_argument(
        "--port", "-p",
        action="store",
        default=80
    )
    parser.add_argument(
        "--nginx-name",
        action="store",
        default="nginx"
    )
    parser.add_argument(
        "--env-file_path",
        action="store",
        default=".env"
    )
    return parser.parse_args()


def update_conf(conf_path: str, conf_contents: str) -> int:
    with open(conf_path, "w") as file:
        return file.write(conf_contents)


def reload_nginx(conf_path: str, container_name: str = "nginx") -> None:
    print("Copying Nginx configuration to the container...")
    result = subprocess.run(
        ["docker", "cp", conf_path, f"{container_name}:/etc/nginx/conf.d/default.conf"],
        capture_output=True
    )
    print(result.stdout or result.stderr)

    print("Reloading Nginx to apply the new configuration...")
    result = subprocess.run(
        ["docker", "exec", container_name, "nginx", "-s", "reload"],
        capture_output=True
    )
    print(result.stdout or result.stderr)

    return print("Nginx configuration updated and reloaded successfully.")


if __name__ == "__main__":
    args: Args = parse_args()

    content = conf_file.format(
      app_hostname=args.app_hostname,
      model_hostname=args.model_hostname,
      port=int(args.port)
    )
    print(update_conf(NGINX_CONF_PATH, content))
    reload_nginx(NGINX_CONF_PATH, args.nginx_name)
