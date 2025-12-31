pipeline {
    agent any

    options{
        // Max number of build logs to keep and days to keep
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        // Enable timestamp at each job in the pipeline
        timestamps()
    }

    environment{
        registry = 'lethanhhiep0220/ahs'
        registryCredential = 'dockerhub'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            agent {
                docker {
                    image 'python:3.10' 
                }
            }
            steps {
                echo 'Testing model correctness..'
                // sh 'pytest'
            }
        }
        stage('Build') {
            steps {
                script {
                    echo 'Building image for deployment...'
                    // Specify Dockerfile path (-f) and build context (.) because Dockerfile is in docker/
                    dockerImage = docker.build("${registry}:${BUILD_NUMBER}", "-f docker/Dockerfile .")
                    echo 'Pushing image to dockerhub...'
                    docker.withRegistry('', registryCredential) {
                        dockerImage.push('latest')
                        // dockerImage.push(GIT_COMMIT_SHORT)
                    }
                }
            }
        }
        stage('Deploy to GKE') {
            steps {
                echo 'Deploying models..'
                echo 'Running a script to trigger pull and start a docker container.'
            }
        }
    }
}
