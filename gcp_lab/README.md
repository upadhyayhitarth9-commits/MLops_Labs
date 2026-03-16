# GCP Lab — Cloud Run with Flask, Cloud Storage & BigQuery

A containerized Python Flask application deployed on Google Cloud Run that integrates with Google Cloud Storage and BigQuery.

---

## What This App Does

This app is a web service with 3 endpoints:

| Endpoint | Description |
|----------|-------------|
| `/` | Returns a hello message — confirms the app is live |
| `/upload` | Uploads a `hello.txt` file to a Google Cloud Storage bucket |
| `/query` | Queries the BigQuery public dataset and returns the top 10 names in Texas |

---

## Project Structure

```
gcp-lab/
├── app.py               # Flask application with 3 routes
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container configuration
└── README.md            # Project documentation
```

---

## Tech Stack

- **Python 3.9** — Application runtime
- **Flask** — Web framework
- **Gunicorn** — Production WSGI server
- **Docker** — Containerization
- **Google Cloud Run** — Serverless deployment platform
- **Google Cloud Storage** — File storage
- **BigQuery** — Data warehouse querying

---

## Prerequisites

- Google Cloud Account with billing enabled
- Google Cloud SDK (`gcloud`) installed
- Docker Desktop installed and running
- Python 3.9+

---

## Setup & Deployment

### 1. Install & Configure gcloud

```bash
# Download gcloud for Mac (M1/M2/M3)
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-arm.tar.gz
tar -xf google-cloud-cli-darwin-arm.tar.gz
./google-cloud-sdk/install.sh

# Add to PATH
source ~/google-cloud-sdk/path.zsh.inc

# Login
gcloud auth login
```

### 2. Create GCP Project

```bash
gcloud projects create gcp-lab-hitarth --name="GCP Lab"
gcloud config set project gcp-lab-hitarth
```

### 3. Enable Required APIs

```bash
gcloud services enable run.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  containerregistry.googleapis.com
```

> ⚠️ Billing must be enabled on your project before this step.

### 4. Create Cloud Storage Bucket

```bash
gsutil mb -l us-central1 gs://hitarth-gcp-lab-bucket
```

### 5. Create Service Account & Assign Roles

```bash
# Create service account
gcloud iam service-accounts create cloud-run-sa \
  --display-name="Cloud Run Service Account"

# Assign Storage Admin role
gcloud projects add-iam-policy-binding gcp-lab-hitarth \
  --member="serviceAccount:cloud-run-sa@gcp-lab-hitarth.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Assign BigQuery User role
gcloud projects add-iam-policy-binding gcp-lab-hitarth \
  --member="serviceAccount:cloud-run-sa@gcp-lab-hitarth.iam.gserviceaccount.com" \
  --role="roles/bigquery.user"
```

### 6. Build & Push Docker Image

```bash
# Authenticate Docker with GCR
gcloud auth configure-docker

# Build for linux/amd64 (required for Cloud Run)
docker build --platform linux/amd64 -t gcr.io/gcp-lab-hitarth/cloud-run-app .

# Push to Container Registry
docker push gcr.io/gcp-lab-hitarth/cloud-run-app
```

> ⚠️ Always use `--platform linux/amd64` on M1/M2/M3 Macs.

### 7. Deploy to Cloud Run

```bash
gcloud run deploy cloud-run-service \
  --image gcr.io/gcp-lab-hitarth/cloud-run-app \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --update-env-vars BUCKET_NAME=hitarth-gcp-lab-bucket \
  --service-account cloud-run-sa@gcp-lab-hitarth.iam.gserviceaccount.com
```

---

## Testing

```bash
# Get your service URL
gcloud run services describe cloud-run-service \
  --platform managed \
  --region us-central1 \
  --format "value(status.url)"

# Test root endpoint
curl https://cloud-run-service-974272331800.us-central1.run.app/

# Test file upload to GCS
curl https://cloud-run-service-974272331800.us-central1.run.app/upload

# Test BigQuery query
curl https://cloud-run-service-974272331800.us-central1.run.app/query

# Verify file was uploaded to bucket
gsutil ls gs://hitarth-gcp-lab-bucket/
```

### Expected Outputs

```
/       → Hello from the intermediate lab!
/upload → File uploaded to hitarth-gcp-lab-bucket.
/query  → Top names in Texas: James, John, Michael, David, Robert, Mary...
```

---

## Monitoring

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **Cloud Run** → click **cloud-run-service**
3. **Logs tab** — view all incoming requests
4. **Observability tab** → **Metrics** — view CPU, memory, request count, latency

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `BUCKET_NAME` | Name of your Cloud Storage bucket |
| `PORT` | Port for the Flask app (default: 8080) |

---

## Clean Up

To avoid charges, delete all resources after the lab:

```bash
# Delete Cloud Run service
gcloud run services delete cloud-run-service --region us-central1

# Delete container image
gcloud container images delete \
  gcr.io/gcp-lab-hitarth/cloud-run-app --force-delete-tags

# Delete storage bucket
gsutil rm -r gs://hitarth-gcp-lab-bucket

# Delete service account
gcloud iam service-accounts delete \
  cloud-run-sa@gcp-lab-hitarth.iam.gserviceaccount.com
```

---

## Architecture

```
User (curl / browser)
        ↓
Google Cloud Run (Flask + Gunicorn)
        ↓                    ↓
Cloud Storage          BigQuery
(upload file)       (query public dataset)
```