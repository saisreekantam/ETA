#!/usr/bin/env bash
# One-time bring-up: everything on Google Cloud, on a single GKE cluster.
#
# Creates a GKE cluster with a small CPU pool (Postgres/frontend/misc) plus a
# spot/preemptible GPU pool (Ollama LLM + RT-DETR vision inference), installs the
# NVIDIA device plugin and ingress-nginx, then applies deploy/k8s/overlays/gke.
#
# Prereqs: `gcloud` installed and authenticated (`gcloud auth login`), a GCP project
# with billing enabled, and the GPU quota for your chosen zone/machine type approved
# (L4 quota is usually instant; ask GCP support to raise it if the pool creation fails
# with a quota error -- this is the #1 first-run snag).
#
# Usage: PROJECT_ID=my-project ./deploy/gcp/setup-gke.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID=your-gcp-project}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
CLUSTER="${CLUSTER:-eta-safety}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
GPU_MACHINE="${GPU_MACHINE:-g2-standard-8}"   # 1x L4, 8 vCPU, 32GB -- fits qwen2.5:14b (Q4) comfortably

echo "==> Project: $PROJECT_ID   Region/zone: $REGION/$ZONE   Cluster: $CLUSTER"
gcloud config set project "$PROJECT_ID"

echo "==> Enabling required APIs (idempotent)"
gcloud services enable container.googleapis.com compute.googleapis.com artifactregistry.googleapis.com

if ! gcloud container clusters describe "$CLUSTER" --zone "$ZONE" >/dev/null 2>&1; then
  echo "==> Creating cluster with a small default (CPU) node pool"
  gcloud container clusters create "$CLUSTER" \
    --zone "$ZONE" \
    --num-nodes 2 \
    --machine-type e2-standard-4 \
    --release-channel regular
else
  echo "==> Cluster $CLUSTER already exists -- reusing it"
fi

gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE"

if ! gcloud container node-pools describe gpu-pool --cluster "$CLUSTER" --zone "$ZONE" >/dev/null 2>&1; then
  echo "==> Creating the GPU node pool (spot, scales to 0 when idle -- Ollama/vision are the only GPU consumers)"
  gcloud container node-pools create gpu-pool \
    --cluster "$CLUSTER" --zone "$ZONE" \
    --machine-type "$GPU_MACHINE" \
    --accelerator "type=$GPU_TYPE,count=1" \
    --spot \
    --num-nodes 1 \
    --enable-autoscaling --min-nodes 0 --max-nodes 2
else
  echo "==> GPU pool already exists -- reusing it"
fi

echo "==> Installing the NVIDIA device plugin (GKE-managed DaemonSet)"
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

echo "==> Installing ingress-nginx (skip if you already run one on this cluster)"
if ! kubectl get ns ingress-nginx >/dev/null 2>&1; then
  helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx --force-update
  helm install ingress-nginx ingress-nginx/ingress-nginx -n ingress-nginx --create-namespace
fi

echo "==> Applying the app (deploy/k8s/overlays/gke)"
kubectl apply -k "$(dirname "$0")/../k8s/overlays/gke"

echo "==> Waiting for the ingress-nginx LoadBalancer IP..."
kubectl -n ingress-nginx get svc ingress-nginx-controller -w &
WATCH_PID=$!
sleep 30 && kill "$WATCH_PID" 2>/dev/null || true

cat <<'EOF'

Done. Next steps:
  1. Get the external IP:  kubectl -n ingress-nginx get svc ingress-nginx-controller
  2. Point your domain's A record at it (or edit deploy/k8s/overlays/gke/kustomization.yaml's
     ingress host patch to your domain and re-apply), or just curl the IP with a Host header
     to smoke-test before DNS propagates.
  3. Watch first boot (migrations + demo seed + ~5GB model pull):
       kubectl -n industrial-safety logs -f deploy/backend
       kubectl -n industrial-safety logs -f deploy/ollama
EOF
