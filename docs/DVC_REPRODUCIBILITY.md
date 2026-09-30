# Phase 4: DVC Reproducibility Guide
## Smart Factory Predictive Maintenance MLOps Pipeline

This document details the Data Version Control (DVC) architecture, pipeline structure, remote storage configuration, and reproducibility procedures for the Smart Factory Predictive Maintenance system.

---

### 1. DVC Pipeline Architecture

The pipeline is organized into 4 distinct, modular stages defined in [`dvc.yaml`](file:///c:/Users/Sowmya%20damodaran/Downloads/MLOPS/smart_factory_pdm/dvc.yaml). Each stage consumes explicit dependencies, reads parameters from [`configs/config.yaml`](file:///c:/Users/Sowmya%20damodaran/Downloads/MLOPS/smart_factory_pdm/configs/config.yaml), and produces version-controlled outputs tracked in [`dvc.lock`](file:///c:/Users/Sowmya%20damodaran/Downloads/MLOPS/smart_factory_pdm/dvc.lock).

```
         +---------+      
         | prepare |      
         +---------+      
              *           
              *           
              *           
        +-----------+     
        | featurize |     
        +-----------+     
         **        **     
       **            *    
      *               **  
+-------+               * 
| train |             **  
+-------+            *    
         **        **     
           **    **       
             *  *         
        +----------+      
        | evaluate |      
        +----------+      
```

---

### 2. Stage Breakdown

#### Stage 1: `prepare`
* **Command**: `python src/stages/prepare.py`
* **Dependencies**:
  * Raw datasets: `data/raw/PdM_telemetry.csv`, `PdM_errors.csv`, `PdM_failures.csv`, `PdM_maint.csv`, `PdM_machines.csv`
  * Source code: `src/stages/prepare.py`, `src/data`
* **Parameters**: `configs/config.yaml: feature_engineering.failure_label_window_hours`
* **Outputs**:
  * `data/interim/merged_dataset.csv`
  * `data/interim/cleaned_dataset.csv`

#### Stage 2: `featurize`
* **Command**: `python src/stages/featurize.py`
* **Dependencies**:
  * `data/interim/cleaned_dataset.csv`
  * Source code: `src/stages/featurize.py`, `src/features`
* **Parameters**:
  * `feature_engineering.rolling_window_hours`
  * `feature_engineering.lag_hours`
  * `feature_engineering.error_window_hours`
  * `preprocessing.split_strategy`
  * `preprocessing.split_ratio`
* **Outputs**:
  * `data/processed/features_engineered.csv`
  * `data/processed/train_data.csv`
  * `data/processed/test_data.csv`

#### Stage 3: `train`
* **Command**: `python src/stages/train.py`
* **Dependencies**:
  * `data/processed/train_data.csv`
  * Source code: `src/stages/train.py`, `src/models`
* **Parameters**:
  * `models.xgboost.n_estimators`
  * `models.xgboost.max_depth`
  * `models.xgboost.learning_rate`
  * `models.xgboost.subsample`
  * `models.xgboost.colsample_bytree`
  * `models.xgboost.scale_pos_weight`
* **Outputs**:
  * `models/saved_models/best_model.joblib` (XGBoost 2.0.0, 31 features)
  * `models/saved_models/feature_names.pkl`
  * `models/saved_models/label_encoder.pkl`
  * `models/saved_models/preprocessor.pkl`
  * `models/saved_models/model_manifest.json`

#### Stage 4: `evaluate`
* **Command**: `python src/stages/evaluate.py`
* **Dependencies**:
  * `models/saved_models/best_model.joblib`
  * `models/saved_models/feature_names.pkl`
  * `data/processed/test_data.csv`
  * Source code: `src/stages/evaluate.py`, `src/evaluation`
* **Parameters**: `evaluation.primary_metric`
* **Metrics / Reports**:
  * `reports/metrics/evaluation_metrics.csv`
  * `reports/metrics/model_comparison.csv`

---

### 3. Remote Storage Configuration

A default local DVC remote has been configured:
* **Remote Name**: `local_storage`
* **Location**: `backups/dvc_remote`
* **Configuration File**: [`.dvc/config`](file:///c:/Users/Sowmya%20damodaran/Downloads/MLOPS/smart_factory_pdm/.dvc/config)

```ini
[core]
    remote = local_storage
['remote "local_storage"']
    url = ../backups/dvc_remote
```

All 10 pipeline outputs (datasets, models, metadata) are synchronized to the remote storage and can be pulled or pushed across environments.

---

### 4. Operational Commands Cheat Sheet

#### 1. Inspect Pipeline Status
```bash
python -m dvc status
```
*Outputs: `Data and pipelines are up to date.`*

#### 2. Visualize Pipeline DAG
```bash
python -m dvc dag
```

#### 3. Verify Remote Synchronization
```bash
python -m dvc status -r local_storage
```
*Outputs: `Cache and remote 'local_storage' are in sync.`*

#### 4. Push / Pull from Remote
```bash
# Push tracked artifacts to remote
python -m dvc push

# Pull artifacts from remote into workspace
python -m dvc pull
```

#### 5. Non-Destructive Reproduction Check
```bash
python -m dvc repro --dry
```
*Skips stages whose inputs, code, and parameters have not changed.*
