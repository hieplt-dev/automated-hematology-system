# Use the official Jenkins base image
FROM jenkins/jenkins:lts

# Set the user to root to install packages
USER root

RUN apt-get update && \
    apt-get install -y git maven && \
    rm -rf /var/lib/apt/lists/*

# Switch back to the jenkins user
USER jenkins

# Expose the Jenkins web interface port and JNLP agent port
EXPOSE 8080
EXPOSE 50000