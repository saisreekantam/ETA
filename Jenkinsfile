// Jenkins alternative to .github/workflows/deploy.yml -- same three stages, for teams
// running an in-house Jenkins instead of GitHub Actions. Expects two credentials:
//   registry-creds : username/password for your image registry
//   kubeconfig     : secret file with cluster access
// and a REGISTRY env var (e.g. registry.example.com/safety) set on the job/agent.
pipeline {
    agent any

    environment {
        BACKEND_IMAGE  = "${env.REGISTRY ?: 'ghcr.io/saisreekantam'}/eta-backend"
        FRONTEND_IMAGE = "${env.REGISTRY ?: 'ghcr.io/saisreekantam'}/eta-frontend"
        TAG = "${env.GIT_COMMIT?.take(12) ?: 'dev'}"
    }

    stages {
        stage('Checks') {
            steps {
                sh 'python3 -m compileall -q server agents models rag db simulator vision eval scripts'
                sh 'cd frontend && npm ci && npx vite build'
            }
        }

        stage('Build & push images') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'registry-creds',
                                                   usernameVariable: 'REG_USER',
                                                   passwordVariable: 'REG_PASS')]) {
                    sh '''
                        echo "$REG_PASS" | docker login -u "$REG_USER" --password-stdin "${REGISTRY%%/*}"
                        docker build -t "$BACKEND_IMAGE:$TAG" -t "$BACKEND_IMAGE:latest" .
                        docker build -t "$FRONTEND_IMAGE:$TAG" -t "$FRONTEND_IMAGE:latest" \
                            --build-arg VITE_API_BASE=/api ./frontend
                        docker push "$BACKEND_IMAGE:$TAG";  docker push "$BACKEND_IMAGE:latest"
                        docker push "$FRONTEND_IMAGE:$TAG"; docker push "$FRONTEND_IMAGE:latest"
                    '''
                }
            }
        }

        stage('Deploy') {
            when { branch 'main' }
            steps {
                withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
                    sh '''
                        kubectl apply -k deploy/k8s/base
                        kubectl -n industrial-safety set image deployment/backend  backend="$BACKEND_IMAGE:$TAG"
                        kubectl -n industrial-safety set image deployment/frontend frontend="$FRONTEND_IMAGE:$TAG"
                        kubectl -n industrial-safety rollout status deployment/backend  --timeout=10m
                        kubectl -n industrial-safety rollout status deployment/frontend --timeout=5m
                    '''
                }
            }
        }
    }
}
