FROM jenkins/jenkins:lts

USER root

# Install docker client (docker.io) from Debian repo and Docker Compose v2 CLI plugin
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl lsb-release apt-transport-https gnupg \
    docker.io \
  && mkdir -p /usr/local/lib/docker/cli-plugins \
  && curl -SL "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-linux-x86_64" \
     -o /usr/local/lib/docker/cli-plugins/docker-compose \
  && chmod +x /usr/local/lib/docker/cli-plugins/docker-compose \
  && rm -rf /var/lib/apt/lists/*

# run as jenkins user
USER jenkins