# Smart Factory PDM — Production Deployment & Operations Guide

> **Version:** 2.0.0 | **Last Updated:** 2026-08-06

This guide covers every deployment path for the Smart Factory Predictive Maintenance platform: local development, Docker Compose production, and enterprise cloud environments.

---

## 📋 Component Version Matrix

| Component | Version | Location |
|:---|:---|:---|
| **Platform** | `2.0.0` | `VERSION` |
| **API Version** | `v2.0.0` | `/api/*` REST endpoints |
| **Backend Service** | `v2.0.0` | `backend/app.py` → Gunicorn |
| **Frontend Dashboard** | `v1.0.0` | `frontend/` → React + Vite + Nginx |
| **Active ML Model** | `v2.0.0` | `models/saved_models/best_model.joblib` |
| **MLflow Registry** | `v2.0.0` | `sqlite:///mlflow.db` |

---

## 🏗️ Architecture Overview

```
Internet / Users
       │
       ▼
┌──────────────────┐
│  Nginx (port 80) │  ← Static React dashboard + Reverse proxy
└─────────┬────────┘
          │  /api/* proxied to backend
          ▼
┌──────────────────┐          ┌──────────────────────┐
│  Flask + Gunicorn│ ────────▶│  MLflow Server       │
│  (port 5000)     │          │  (port 5001)         │
└─────────┬────────┘          └──────────┬───────────┘
          │                              │
          ▼                              ▼
┌──────────────────┐          ┌──────────────────────┐
│  SQLite DB       │          │  MLflow SQLite DB    │
│  (predictions)   │          │  + Artifact Store    │
└──────────────────┘          └──────────────────────┘

All services communicate on the internal `pdm_network` Docker bridge.
```

---

## ✅ Prerequisites

### For Docker Deployment
- Docker Engine `20.10+`
- Docker Compose `v2.0+` (included with Docker Desktop)
- 4 GB RAM minimum (8 GB recommended for ML model loading)
- 10 GB free disk space

```bash
# Verify versions
docker --version        # Docker version 24.x.x
docker compose version  # Docker Compose version v2.x.x
```

### For Local Development
- Python `3.10+`
- Node.js `18+`
- pip and npm

---

## 🚀 Option 1: Docker Compose (Recommended for Production)

### Step 1 — Clone and configure
```bash
git clone https://github.com/your-org/smart_factory_pdm.git
cd smart_factory_pdm

# Create your local .env file from the template
cp .env.example .env
```

### Step 2 — Edit .env
Open `.env` and update these critical values:

```bash
# Generate a strong secret key:
python -c "import secrets; print(secrets.token_hex(32))"

FLASK_SECRET_KEY=<paste-generated-key-here>
ALLOWED_ORIGINS=https://your-domain.com    # or http://localhost for local
```

### Step 3 — Build and launch
```bash
# Production mode (uses docker-compose.yml only, no override)
docker-compose -f docker-compose.yml up --build -d
```

### Step 4 — Verify all services are healthy
```bash
# Check container status and health
docker-compose ps

# Or use the built-in health check script
bash scripts/healthcheck.sh
```

### Step 5 — Access the platform
| Service | URL |
|---|---|
| 🖥️ **Frontend Dashboard** | http://localhost |
| ⚡ **Backend API** | http://localhost:5000 |
| 🧬 **MLflow Tracking UI** | http://localhost:5001 |
| 💓 **Backend Health** | http://localhost:5000/health |
| 🟢 **Nginx Health** | http://localhost/nginx-health |

---

## 🛠️ Option 2: Local Development

### Backend (Flask dev server with hot reload)
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate         # Windows
source venv/bin/activate         # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install gunicorn psutil evidently

# Set environment variables
cp .env.development .env

# Run Flask development server (auto-reload)
python backend/app.py
# → http://localhost:5000
```

### Frontend (Vite dev server)
```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### MLflow Server
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001
# → http://localhost:5001
```

---

## 🐳 Option 3: Docker Compose Development Mode

The `docker-compose.override.yml` file auto-merges for development with hot reload:

```bash
# Development mode (merges docker-compose.yml + docker-compose.override.yml)
cp .env.development .env
docker-compose up --build

# Backend source changes are reflected immediately (mounted as volumes)
# Frontend changes require a browser refresh
```

---

## ☁️ Option 4: Cloud Deployment

### AWS ECS / Fargate

#### 1. Build and push images to ECR
```bash
AWS_ACCOUNT_ID=123456789012
AWS_REGION=us-east-1
ECR_BASE=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Authenticate
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${ECR_BASE}

# Create ECR repositories (one-time)
for svc in backend frontend mlflow; do
  aws ecr create-repository --repository-name smart-factory-pdm-${svc}
done

# Build and push all images
VERSION=$(cat VERSION)
for svc in backend frontend mlflow; do
  docker build -f Dockerfile.${svc} \
    --build-arg VERSION=${VERSION} \
    --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
    -t ${ECR_BASE}/smart-factory-pdm-${svc}:${VERSION} .
  docker push ${ECR_BASE}/smart-factory-pdm-${svc}:${VERSION}
done
```

#### 2. ECS Task Definition (backend example)
```json
{
  "family": "smart-factory-pdm-backend",
  "cpu": "1024",
  "memory": "2048",
  "networkMode": "awsvpc",
  "containerDefinitions": [{
    "name": "backend",
    "image": "<ECR_URI>/smart-factory-pdm-backend:2.0.0",
    "portMappings": [{"containerPort": 5000}],
    "environment": [
      {"name": "FLASK_ENV", "value": "production"},
      {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow-service:5001"}
    ],
    "secrets": [
      {"name": "FLASK_SECRET_KEY", "valueFrom": "arn:aws:secretsmanager:..."}
    ],
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -sf http://localhost:5000/health || exit 1"],
      "interval": 30,
      "timeout": 10,
      "retries": 3,
      "startPeriod": 30
    }
  }]
}
```

---

### Azure Container Apps

```bash
RESOURCE_GROUP=smart-factory-rg
LOCATION=eastus
ENVIRONMENT=smart-factory-env

# Create environment
az containerapp env create \
  --name ${ENVIRONMENT} \
  --resource-group ${RESOURCE_GROUP} \
  --location ${LOCATION}

# Deploy backend
az containerapp create \
  --name pdm-backend \
  --resource-group ${RESOURCE_GROUP} \
  --environment ${ENVIRONMENT} \
  --image ghcr.io/your-org/smart-factory-pdm-backend:2.0.0 \
  --target-port 5000 \
  --ingress internal \
  --min-replicas 1 \
  --max-replicas 5 \
  --cpu 1.0 \
  --memory 2Gi \
  --env-vars FLASK_ENV=production FLASK_DEBUG=0
```

---

### GCP Cloud Run

```bash
PROJECT_ID=your-gcp-project
REGION=us-central1

# Submit backend image to Artifact Registry
gcloud builds submit \
  --tag gcr.io/${PROJECT_ID}/smart-factory-pdm-backend:2.0.0 \
  --file Dockerfile.backend .

# Deploy to Cloud Run
gcloud run deploy smart-factory-pdm-backend \
  --image gcr.io/${PROJECT_ID}/smart-factory-pdm-backend:2.0.0 \
  --region ${REGION} \
  --platform managed \
  --port 5000 \
  --memory 2Gi \
  --concurrency 80 \
  --set-env-vars FLASK_ENV=production,FLASK_DEBUG=0
```

---

## 🔐 Environment Variable Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `FLASK_SECRET_KEY` | ✅ **Yes** | — | Flask session signing key. Generate with `secrets.token_hex(32)` |
| `FLASK_DEBUG` | No | `0` | Set `1` for dev only. NEVER `1` in production |
| `FLASK_ENV` | No | `production` | `production` or `development` |
| `FLASK_PORT` | No | `5000` | Port Gunicorn binds to |
| `WEB_CONCURRENCY` | No | `(2×CPU)+1` | Number of Gunicorn worker processes |
| `GUNICORN_THREADS` | No | `2` | Threads per worker |
| `GUNICORN_TIMEOUT` | No | `120` | Worker silence timeout (seconds) |
| `ALLOWED_ORIGINS` | No | `http://localhost,...` | Comma-separated CORS origins |
| `MLFLOW_TRACKING_URI` | No | `http://mlflow:5001` | MLflow server URL |
| `DB_PATH` | No | `/app/data/predictions.db` | SQLite predictions database path |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity: `DEBUG/INFO/WARNING/ERROR` |
| `BACKUP_RETENTION_DAYS` | No | `7` | Number of backup archives to retain |
| `INSTALL_CLOUD_DEPS` | No | `false` | Install boto3/azure/gcp in MLflow container |
| `APP_VERSION` | No | `2.0.0` | Image tag for docker-compose |

---

## 💾 Backup & Restore

### Create a Backup
```bash
# Production backup (with default 7-archive retention)
python scripts/backup.py

# Keep last 14 archives
python scripts/backup.py --retention 14

# Preview without writing (dry-run)
python scripts/backup.py --dry-run
```

Archives are saved to `backups/backup_pdm_YYYYMMDD_HHMMSS.tar.gz` with MD5 checksums.

### Scheduled Backup (cron)
```bash
# Add to crontab: daily backup at 2:00 AM
0 2 * * * /path/to/venv/bin/python /path/to/smart_factory_pdm/scripts/backup.py >> /var/log/pdm-backup.log 2>&1
```

### Restore from Backup
```bash
# List available backups
python scripts/restore.py --list

# Restore most recent backup
python scripts/restore.py --latest

# Restore specific archive
python scripts/restore.py --file backup_pdm_20240101_120000.tar.gz

# Preview restore without extracting
python scripts/restore.py --latest --dry-run
```

### Backup Contents
Each archive contains:
- `data/predictions.db` — SQLite predictions and request logs
- `mlflow.db` — MLflow experiment tracking database
- `models/saved_models/` — Trained ML model artifacts
- `reports/` — EDA plots, metrics CSVs, validation JSONs
- `mlruns/` — MLflow artifact files (confusion matrices, feature charts)

---

## 🔍 Health Monitoring

### Quick Health Check
```bash
# Human-readable status report
bash scripts/healthcheck.sh

# JSON output (for monitoring tools)
bash scripts/healthcheck.sh --json

# Check individual endpoints
curl -sf http://localhost:5000/health | python -m json.tool
curl -sf http://localhost:5001/health
curl -sf http://localhost/nginx-health
```

### Docker Health Status
```bash
docker-compose ps                    # Show health for all containers
docker inspect pdm_backend | python -m json.tool | grep -A5 Health
```

---

## 📊 Version Information

```bash
# Human-readable version matrix
python scripts/version_info.py

# JSON format for automation
python scripts/version_info.py --json
```

---

## 🔒 Security Hardening

| Feature | Status | Detail |
|---|---|---|
| Non-root execution | ✅ | Backend runs as `appuser` (UID 1000) |
| Non-root MLflow | ✅ | MLflow runs as `mlflow` user |
| Debug mode off | ✅ | `FLASK_DEBUG=0` enforced in production |
| CORS restricted | ✅ | `ALLOWED_ORIGINS` whitelist via env var |
| Security headers | ✅ | X-Frame, CSP, X-Content-Type, Permissions-Policy |
| Server version hidden | ✅ | `server_tokens off` in Nginx |
| Log rotation | ✅ | `json-file` driver with `max-size: 20m, max-file: 5` |
| Secret scanning | ✅ | pip-audit + npm audit in CI/CD pipeline |
| Health checks | ✅ | All containers have Docker HEALTHCHECK directives |

---

## 🐛 Troubleshooting

### Container fails to start
```bash
docker-compose logs backend       # View backend logs
docker-compose logs mlflow        # View MLflow logs
docker-compose logs frontend      # View Nginx logs
```

### Backend health check failing
```bash
# Check if the ML model files are present
docker-compose exec backend ls -la /app/models/saved_models/

# Check DB path
docker-compose exec backend ls -la /app/data/

# Test health endpoint directly
docker-compose exec backend curl -sf http://localhost:5000/health
```

### MLflow database locked
```bash
# Stop all services first
docker-compose down

# Restart in correct order (mlflow → backend → frontend)
docker-compose up -d mlflow
sleep 10
docker-compose up -d backend
sleep 15
docker-compose up -d frontend
```

### Port already in use
```bash
# Check which process is using port 5000
netstat -ano | findstr :5000       # Windows
lsof -i :5000                      # macOS/Linux

# Change the port in .env
BACKEND_PORT=5010
docker-compose up -d
```

### Out of disk space (Docker volumes)
```bash
# Check volume sizes
docker system df

# Clean up dangling images and stopped containers
docker system prune

# Remove a specific unused volume
docker volume rm pdm_data          # ⚠️ Destructive!
```

---

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline (`.github/workflows/ci-cd.yml`) runs on every push to `main`, `master`, and `develop` branches:

| Job | Trigger | Description |
|---|---|---|
| `backend-quality` | All pushes | Flake8 lint + smoke test + pytest coverage |
| `frontend-quality` | All pushes | TypeScript check + Vite production build |
| `security-scan` | All pushes | pip-audit + npm audit (advisory, non-blocking) |
| `docker-build-push` | After quality jobs pass | Build all 3 images, push to `ghcr.io` on main |
| `release` | `v*.*.*` tags only | Create GitHub Release with changelogs |

### Publish a Release
```bash
# Update VERSION file
echo "2.1.0" > VERSION

# Commit and tag
git add VERSION
git commit -m "chore: bump version to 2.1.0"
git tag v2.1.0
git push origin main --tags

# CI/CD will automatically:
# 1. Build and push images tagged :2.1.0 and :latest to ghcr.io
# 2. Create a GitHub Release with changelog
```

---

## ⚡ Performance Reference

| Optimization | Implementation |
|---|---|
| Multi-stage Docker builds | All 3 Dockerfiles use builder → runtime pattern |
| `.dockerignore` | Excludes node_modules, data CSVs, __pycache__, venv |
| Gunicorn `preload_app` | ML model loaded once in master, shared via fork COW |
| `max_requests + jitter` | Workers recycle after 1000 requests to prevent memory bloat |
| Vite vendor chunk splitting | React, Recharts, Framer Motion in separate cached bundles |
| Nginx gzip level 6 | All text assets compressed before transfer |
| Nginx 1-year immutable caching | Content-hashed static assets cached forever |
| Nginx upstream keepalive | 32 persistent connections to Gunicorn pool |
| Docker log rotation | `max-size: 20m, max-file: 5` per service |
| Resource limits | CPU/memory limits prevent container resource starvation |
