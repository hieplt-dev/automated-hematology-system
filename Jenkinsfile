pipeline {
    agent any

    options{
        // Max number of build logs to keep and days to keep
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        // Enable timestamp at each job in the pipeline
        timestamps()
    }

    environment{
        // ===== Docker Hub =====
        IMAGE_REGISTRY        = 'lethanhhiep0220/ahs'
        REGISTRY_CREDENTIAL   = 'dockerhub'

        // ===== GCP / GKE =====
        GCP_PROJECT   = 'ahsys-480510'
        GKE_CLUSTER   = 'ahsys-480510-gke'
        GKE_ZONE      = 'asia-southeast2'
        K8S_NAMESPACE = 'model-serving'

        // ===== Helm =====
        HELM_RELEASE = 'hematology-api'
        HELM_CHART   = 'helm/apps/hematology-api'

        // ===== Model config (PASS TO HELM, NOT PULLED HERE) =====
        MODEL_BUCKET = 'ahsys-480510-model-registry'
        MODEL_NAME   = 'hematology-model'
        MODEL_STAGE  = 'production'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            agent { docker { image 'python:3.10' } }
            steps {
                echo 'Testing model correctness..'
                // sh 'pytest'
            }
        }

        stage('Build & Push Image') {
            steps {
                script {
                    echo 'Building image for deployment..'
                    dockerImage = docker.build(IMAGE_REGISTRY   + ":$BUILD_NUMBER", "-f docker/Dockerfile .")
                    echo 'Pushing image to dockerhub..'
                    docker.withRegistry( '', REGISTRY_CREDENTIAL ) {
                        dockerImage.push("$BUILD_NUMBER")
                        dockerImage.push('latest')
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

        stage('Prepare GCP Secret') {
            steps {
                sh '''
                kubectl get secret gcp-sa-key -n model-serving >/dev/null 2>&1 || \
                kubectl create secret generic gcp-sa-key \
                    --from-file=$GOOGLE_APPLICATION_CREDENTIALS \
                    -n model-serving
                '''
            }
        }

        stage('Deploy Monitoring with Helm') {
            steps {
                // take slack api url from jenkins credential
                withCredentials([string(credentialsId: 'slack-api-url', variable: 'SLACK_URL')]) {
                    sh '''
                        helm upgrade --install ${HELM_RELEASE} ${HELM_CHART} \
                          --namespace ${K8S_NAMESPACE} \
                          --create-namespace \
                          --set image.tag=${BUILD_NUMBER} \
                          --set model.bucket=${MODEL_BUCKET} \
                          --set model.name=${MODEL_NAME} \
                          --set model.stage=${MODEL_STAGE} \
                          --set alertmanager.config.global.slack_api_url=${SLACK_URL} \
                          -f helm/monitoring/prometheus/values.yaml
                        
                        kubectl apply -f helm/monitoring/alertmanager/prometheus-rule.yaml -n monitoring

                        kubectl apply -f helm/monitoring/alertmanager/alert-config.yml -n monitoring
                    '''
                }
            }
        }
    }
}
