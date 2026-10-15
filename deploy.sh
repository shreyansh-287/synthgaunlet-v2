#!/usr/bin/env bash
# Deploy SynthGauntlet to Google Cloud Run.
set -eo pipefail

PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="synthgauntlet"
IMAGE_URI="us-central1-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME}:v2.0.14"

if [ -z "${PROJECT_ID}" ]; then
  echo "ERROR: set GCP_PROJECT or run 'gcloud config set project <id>'." >&2
  exit 1
fi

echo "Building image ${IMAGE_URI} ..."
gcloud builds submit --tag "${IMAGE_URI}" .

echo "Deploying to Cloud Run (${REGION}) ..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_URI}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 4Gi --cpu 2 \
  --min-instances "${MIN_INSTANCES:-1}" --max-instances 5 \
  --timeout 600s \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=${GOOGLE_GENAI_USE_VERTEXAI:-true},GEMINI_API_KEY=${GEMINI_API_KEY},GCP_PROJECT=${PROJECT_ID},VERTEX_LOCATION=${VERTEX_LOCATION:-global},GENERATOR_MODEL=gemini-3.6-flash,ATTACKER_MODEL=gemini-3.6-flash,ORCHESTRATOR_MODEL=gemini-3.6-flash,REPORTER_MODEL=gemini-3.6-flash,LLM_TIMEOUT_MS=${LLM_TIMEOUT_MS:-30000},LLM_MAX_RETRIES=${LLM_MAX_RETRIES:-3},GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-845643368139-kjafc7tpu1eus9cfg12jm6gulue4cue3.apps.googleusercontent.com}"

gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" \
  --format="value(status.url)"
