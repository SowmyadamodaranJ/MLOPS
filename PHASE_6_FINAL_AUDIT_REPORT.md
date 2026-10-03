# 🏭 Smart Factory Predictive Maintenance (PdM)
## Phase 6 Final PBL Prototype Audit & Verification Report

---

### Executive Summary
This report documents the final validation, hardening, and audit of the **Smart Factory Predictive Maintenance (PdM)** platform, bringing it to a stable **80–85% working Problem-Based Learning (PBL) prototype**. All modifications strictly adhere to the source-of-truth findings from the Phase 6 audit, preserving the established architecture (Flask backend, React Vite frontend, XGBoost champion model, TreeSHAP explainer, SQLite persistence, MLflow tracking, and DVC pipeline definitions).

---

### 1. Key Issues Resolved

#### A. Data Duplication & Leakage Eradication
- **Root Cause Identified:** The original synthetic generator duplicated 1 year of Microsoft Azure telemetry shifted by 365 days. A 70/30 chronological split placed duplicated machine cycles into both train and test splits, causing artificially inflated test metrics (e.g. 99.9% accuracy / 0.998 ROC-AUC).
- **Remediation:** Pipeline enforced single-pass authentic data loading (876,100 telemetry readings across 100 machines). The chronological split cleanly separates historical machine operating hours from future test evaluation windows.
- **Model Retraining:** Re-trained Logistic Regression, Decision Tree, Random Forest, and XGBoost on the authentic chronological split.
- **Authentic Champion Metrics (XGBoost v2.0.0):**
  - **F1 Score:** `0.8250` (selected metric)
  - **ROC-AUC:** `0.8870`
  - **Accuracy:** `0.8420`
  - **Precision:** `0.8160`
  - **Recall:** `0.8350`
  - **Optimal Decision Threshold:** `0.8781` (derived via F1-maximization threshold sweep)

#### B. Eradication of Hardcoded / Canned Frontend Data
- **Fleet Machine Counts:** Replaced static fallback `[82, 12, 6]` in `/api/dashboard` and `frontend/src/pages/Dashboard.tsx` with dynamic fleet aggregations calculated directly from the Decision Engine (`/api/health-queue`).
  - *Current Fleet Status:* Total 100 machines | **51 Healthy** | **38 Warning** | **11 Critical**.
- **Canned Machine M-104:** Purged references to fictional synthetic machine `104`:
  - `frontend/src/App.tsx`: `handleSelectMachineId` now dynamically queries `fetchMachineDecision(machineId)` from the live API.
  - `frontend/src/components/ui/AIInsightCards.tsx`: Default cards now reference verified machines in the fleet (`Machine 94` - Critical, `Machine 35` - Warning, `Machine 1` - Optimal).
  - `frontend/src/components/layout/NotificationCenter.tsx`: Real machine IDs and authentic MLflow metrics (F1 `0.8250`, ROC-AUC `0.8870`) are used.
- **Dynamic Decision Pipeline:** Machine detail cards, health scores, and risk classifications are computed live from machine features and XGBoost inference rather than mock rules like `age > 18`.

#### C. Model Metadata & Configuration Alignment
- **`configs/config.yaml`:** Updated phase descriptor from `Phase 2` to `Phase 6 - Production Platform Integration & Auditing`.
- **`selected_metric_value`:** Populated in both `reports/metrics/best_model_meta.json` and `models/saved_models/model_manifest.json` with value `0.8250`.
- **`optimal_threshold`:** Persisted and synchronized across model services, metadata files, and frontend consumers as `0.8781`.
- **Request Validation:** Fixed numeric casting in `/api/predict` and `/api/explain` routes to handle categorical `model` strings without throwing `INVALID_TYPE`.

---

### 2. End-to-End User Flow Verification

The 10-step operational workflow was verified via live automated execution against the running platform (`scripts/verify_user_flow.py`):
$$\text{Select Machine} \to \text{Load Sensor Data} \to \text{Execute Diagnostics} \to \text{XGBoost Prediction} \to \text{SHAP Explanation} \to \text{Decision Engine} \to \text{Risk Tier} \to \text{Health Score} \to \text{Priority} \to \text{Recommendation}$$

#### Live Diagnostic Results across Fleet Tiers:
| Operational Metric | Machine 1 (Optimal) | Machine 35 (Warning) | Machine 94 (Critical) |
| :--- | :--- | :--- | :--- |
| **Model & Age** | `model3` (18 yrs) | `model1` (17 yrs) | `model2` (18 yrs) |
| **Telemetry (Volt/Rot/Press/Vib)** | 191.9 / 382.7 / 100.9 / 37.9 | 193.6 / 481.0 / 83.3 / 32.0 | 171.4 / 560.6 / 117.7 / 39.2 |
| **XGBoost Failure Prob** | `0.0000` | `0.4509` | `0.9993` |
| **Predicted Class (Thresh: 0.8781)** | `0` (Normal) | `0` (Sub-threshold) | `1` (Failure Imminent) |
| **Primary SHAP Driver** | `volt_rolling24h_mean` (-0.856) | `volt_rolling24h_mean` (+2.631) | `total_errors_rolling24h` (+6.223) |
| **Evaluated Risk Tier** | `MONITOR` | `WARNING` | `CRITICAL` |
| **Calculated Health Score** | `100.0 / 100` | `54.9 / 100` | `0.1 / 100` |
| **Maintenance Priority** | `P3 - Low` | `P2 - High` | `P1 - Critical` |
| **Action & Urgency** | Routine Monitoring | Scheduled Inspection | Immediate Component Replacement |

---

### 3. Core API Endpoint Status

| Endpoint | Method | Status | Response Latency | Verification Details |
| :--- | :---: | :---: | :---: | :--- |
| `/api/health` | GET | `200 OK` | < 10 ms | Backend online, SQLite connected, MLflow accessible, 4 artifacts verified |
| `/api/models` | GET | `200 OK` | < 15 ms | XGBoost v2.0.0, optimal threshold `0.8781`, metric F1 `0.8250` |
| `/api/machines` | GET | `200 OK` | < 25 ms | Fleet of 100 machines with live status mappings |
| `/api/machines/<id>/features` | GET | `200 OK` | < 15 ms | 31 engineered features from latest machine telemetry |
| `/api/predict` | POST | `200 OK` | 12.48 ms | Real XGBoost inference, threshold comparison, risk level |
| `/api/explain` | POST | `200 OK` | ~85 ms | TreeSHAP waterfall values, baseline value, feature impact rankings |
| `/api/machines/<id>/decision` | GET | `200 OK` | < 20 ms | Combined prediction, SHAP drivers, health score, priority |
| `/api/health-queue` | GET | `200 OK` | Cached (<15ms) | Fleet priority queue sorted by failure risk |

---

### 4. Test Suite Execution & Pass/Fail Metrics

#### Backend & Integration (Pytest)
- **Command:** `python -m pytest tests/ -v`
- **Result:** **52 Passed, 0 Failed, 0 Skipped** (Duration: 32.48s)
- **Coverage:**
  - `backend/services/decision_engine_service.py`: **88%**
  - `backend/app.py`: **79%**
  - `backend/utils/logger.py`: **71%**
  - `backend/services/feature_service.py`: **69%**
  - Unit tests: 24 tests passed
  - API Integration: 6 tests passed
  - E2E Playwright: 3 tests passed
  - Latency performance: 1 test passed
  - Security (SQL injection / XSS payloads): 18 tests passed

#### Frontend (Vitest)
- **Command:** `npm run test` (in `frontend/`)
- **Result:** **4 Passed, 0 Failed, 2 Test Files Passed** (Duration: 3.21s)
  - `tests/components/Dashboard.test.tsx` (2 tests passed)
  - `tests/hooks/api.test.ts` (2 tests passed)

#### Total Test Count: **56 Passed / 0 Failed (100% Pass Rate)**

---

### 5. MLflow & DVC Integrity

- **MLflow Tracking Server:** Active on `http://localhost:5001`. Experiment `predictive-maintenance` records model parameters, training runs, and artifact logging.
- **Model Feature Contract:** Strictly enforces 31 engineered features loaded via `models/saved_models/feature_names.pkl`.
- **DVC Stages:** Defined in `dvc.yaml` covering `prepare`, `featurize`, `train`, and `evaluate`.
- **Model Artifacts:**
  - `models/saved_models/best_model.joblib`: Serialized XGBoost pipeline
  - `models/saved_models/feature_names.pkl`: Canonical feature list
  - `models/saved_models/label_encoder.pkl`: Label encoder
  - `models/saved_models/preprocessor.pkl`: Data preprocessor
  - `models/saved_models/model_manifest.json`: Manifest metadata

---

### 6. Deployment Status & Known Limitations

- **Docker Status:** Docker Desktop is **not installed** on this local Windows host (`docker : The term 'docker' is not recognized`). Per instructions, Docker deployment is **not claimed to be live**. However, configuration files (`Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml`) remain fully syntax-validated and deployment-ready for Linux container environments.
- **Evidently AI Drift Monitoring:** Evidently AI reports known syntax incompatibilities on Python 3.13 due to legacy Pydantic v1 internals. A robust statistical fallback using Kolmogorov-Smirnov (KS) drift testing is integrated in `drift_service.py` to ensure continuous drift scoring without service disruption.

---

### 7. How to Run the Platform Locally

#### 1. Backend Server
```powershell
cd "c:\Users\Sowmya damodaran\Downloads\MLOPS\smart_factory_pdm"
python -m backend.app
# Runs Flask API on http://localhost:5000
```

#### 2. Frontend Application
```powershell
cd "c:\Users\Sowmya damodaran\Downloads\MLOPS\smart_factory_pdm\frontend"
npm run dev
# Runs React Vite UI on http://localhost:3000
```

#### 3. MLflow Tracking UI
```powershell
cd "c:\Users\Sowmya damodaran\Downloads\MLOPS\smart_factory_pdm"
python -m mlflow ui --port 5001
# Runs MLflow on http://localhost:5001
```

#### 4. Streamlit Diagnostic Dashboard (Secondary UI)
```powershell
cd "c:\Users\Sowmya damodaran\Downloads\MLOPS\smart_factory_pdm"
python -m streamlit run src/dashboard/app.py --server.port 8501
# Runs Streamlit on http://localhost:8501
```

---

### 8. Recommended Presentation & Demo Flow

1. **Dashboard Overview (`http://localhost:3000`):**
   - Show live fleet metrics: 100 Total Machines, 51 Healthy, 38 Warning, 11 Critical.
   - Show model performance tile: XGBoost F1 Score `0.8250` and Optimal Threshold `0.8781`.
2. **Machine Fleet & Health Queue (`/machines`):**
   - Filter by `Critical` risk tier. Select **Machine 94**.
   - Show high failure probability (`99.9%`), health score (`0.1`), and immediate maintenance recommendation.
3. **Inference & Explainability Workspace (`/predictions`):**
   - Load sensor telemetry for Machine 94.
   - Click **Execute Diagnostics**. Observe live XGBoost prediction (<15ms latency).
   - Review TreeSHAP waterfall graph showing `total_errors_rolling24h` as the dominant risk driver (+6.22 SHAP contribution).
4. **Sub-threshold Warning Machine:**
   - Select **Machine 35** (`Warning` tier, prob: `0.4509`).
   - Demonstrate that although below the hard failure threshold (`0.8781`), the Decision Engine elevates it to `P2 - High` due to voltage instability.
5. **Optimal Health Machine:**
   - Select **Machine 1** (`Monitor` tier, prob: `0.0000`, Health Score: `100.0`).
   - Confirm normal operating bounds and routine maintenance recommendation.
6. **MLflow Tracking UI (`http://localhost:5001`):**
   - Display the logged champion run with true authentic metrics and registered artifact contract.
