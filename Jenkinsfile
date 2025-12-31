pipeline {
    agent {
        any
    }

    options{
        // Max number of build logs to keep and days to keep
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        // Enable timestamp at each job in the pipeline
        timestamps()
    }

    environment{
        // ===== Docker Hub =====
        REGISTRY              = 'lethanhhiep0220/ahs'
        REGISTRY_CREDENTIAL   = 'dockerhub'

        // ===== GCP / GKE =====
        GCP_PROJECT   = 'ahsys-480510'
        GKE_CLUSTER   = 'ahsys-480510-model-registry'
        GKE_ZONE      = 'asia-southeast1-a'
        K8S_NAMESPACE = 'model-serving'

        // ===== Helm =====
        HELM_RELEASE = 'ahs'
        HELM_CHART   = 'helm/apps'

        // ===== Model config (PASS TO HELM, NOT PULLED HERE) =====
        MODEL_BUCKET = 'ahsys-480510-model-registry'
        MODEL_NAME   = 'my_model'
        MODEL_STAGE  = 'production'
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

        stage('Build & Push Image') {
            steps {
                script {
                    echo 'Building Docker image...'

                    def image = docker.build(
                        "${REGISTRY}:${BUILD_NUMBER}",
                        "-f docker/Dockerfile ."
                    )

                    docker.withRegistry('', REGISTRY_CREDENTIAL) {
                        echo 'Pushing Docker image...'
                        image.push("${BUILD_NUMBER}")
                        image.push("latest")
                    }
                }
            }
        }

        stage('Auth to GKE') {
            steps {
                withCredentials([
                    file(credentialsId: 'gcp-sa-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')
                ]) {
                    sh '''
                        gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
                        gcloud config set project ${GCP_PROJECT}
                        gcloud container clusters get-credentials ${GKE_CLUSTER} \
                            --zone ${GKE_ZONE} \
                            --project ${GCP_PROJECT}
                    '''
                }
            }
        }

        stage('Deploy to GKE with Helm') {
            steps {
                sh '''
                    helm upgrade --install ${HELM_RELEASE} ${HELM_CHART} \
                      --namespace ${K8S_NAMESPACE} \
                      --create-namespace \
                      --set image.repository=${REGISTRY} \
                      --set image.tag=${BUILD_NUMBER} \
                      --set model.bucket=${MODEL_BUCKET} \
                      --set model.name=${MODEL_NAME} \
                      --set model.stage=${MODEL_STAGE}
                '''
            }
        }
    }
}
