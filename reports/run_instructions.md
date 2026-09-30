# Smart Factory Predictive Maintenance running instructions

Welcome to the Smart Factory Predictive Maintenance dashboard and machine learning execution guide. This document provides step-by-step instructions for executing the ML pipeline and starting the premium Streamlit observability dashboard on your local Windows system.

> [!NOTE]
> All raw datasets (`PdM_telemetry.csv`, `PdM_errors.csv`, `PdM_failures.csv`, `PdM_maint.csv`, `PdM_machines.csv`) are located inside the `smart_factory_pdm/data/raw/` directory, and intermediate processing outputs are already generated.

---

## 1. Direct Execution (easiest & recommended for Windows)

If you get execution policy errors in PowerShell when trying to activate the virtual environment, you can run the commands directly using the paths to the environment's `Scripts/` directory:

### Run MLflow UI
```powershell
# From the project root (c:\Users\Sowmya damodaran\Downloads\MLOPS):
venv\Scripts\mlflow ui --backend-store-uri sqlite:///smart_factory_pdm/mlflow.db --default-artifact-root smart_factory_pdm/mlruns
```

### Run Streamlit Dashboard
```powershell
# From the project root (c:\Users\Sowmya damodaran\Downloads\MLOPS):
venv\Scripts\streamlit run smart_factory_pdm/src/dashboard/app.py
```

### Run ML Pipeline
```powershell
# From the project root (c:\Users\Sowmya damodaran\Downloads\MLOPS):
venv\Scripts\python smart_factory_pdm/run_pipeline.py
```

---

## 2. Environment Activation Method

If you prefer to activate the environment first:

### PowerShell
```powershell
# If script execution is allowed:
venv\Scripts\Activate.ps1
```
*(If you get a script execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, or use the direct method above).*

### Command Prompt (cmd)
```cmd
venv\Scripts\activate.bat
```

Once activated, you can run the commands normally:
```bash
cd smart_factory_pdm
python run_pipeline.py
streamlit run src/dashboard/app.py
mlflow ui
```
