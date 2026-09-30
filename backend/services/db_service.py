"""
db_service.py
-------------
SQLite service for logging predictions, exceptions, API requests, and audit trails.
Updated to support Prediction ID, confidence, latency, model version, and filtered search/export.
"""

import sqlite3
import uuid
import csv
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from backend.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "predictions.db"


class DatabaseService:
    """
    Manages SQLite connection, table creation, prediction audit logs, and metrics search.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Ensure data directory exists
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(DB_PATH)
        self._create_tables()
        self._initialized = True

    def _get_connection(self):
        """Return a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        """Create logging tables if they don't already exist and migrate columns."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # Enhanced Predictions log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT UNIQUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    machine_id INTEGER,
                    volt REAL,
                    rotate REAL,
                    pressure REAL,
                    vibration REAL,
                    prediction INTEGER,
                    probability REAL,
                    confidence REAL,
                    latency_ms REAL,
                    model_version TEXT,
                    risk_level TEXT,
                    recommended_action TEXT
                )
            """)

            # Ensure new columns exist for database upgrades
            cursor.execute("PRAGMA table_info(predictions_log)")
            existing_cols = [row["name"] for row in cursor.fetchall()]
            
            if "prediction_id" not in existing_cols:
                cursor.execute("ALTER TABLE predictions_log ADD COLUMN prediction_id TEXT")
            if "confidence" not in existing_cols:
                cursor.execute("ALTER TABLE predictions_log ADD COLUMN confidence REAL")
            if "latency_ms" not in existing_cols:
                cursor.execute("ALTER TABLE predictions_log ADD COLUMN latency_ms REAL")
            if "model_version" not in existing_cols:
                cursor.execute("ALTER TABLE predictions_log ADD COLUMN model_version TEXT DEFAULT 'v2.0.0'")

            # Exceptions log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exceptions_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    endpoint TEXT,
                    error_message TEXT,
                    stack_trace TEXT
                )
            """)

            # API Requests log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_requests_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    method TEXT,
                    path TEXT,
                    status_code INTEGER,
                    latency_ms REAL
                )
            """)
            conn.commit()
            logger.info("SQLite database tables verified & migrated successfully.")
        except Exception as e:
            logger.error(f"Error creating/migrating SQLite tables: {e}")
        finally:
            conn.close()

    def log_prediction(self, machine_id: int, volt: float, rotate: float, pressure: float, vibration: float,
                       prediction: int, probability: float, risk_level: str, recommended_action: str,
                       confidence: float = 0.0, latency_ms: float = 0.0, model_version: str = "v2.0.0") -> str:
        """Log a detailed model prediction to the database. Returns prediction_id."""
        conn = self._get_connection()
        cursor = conn.cursor()
        pred_id = f"pred_{uuid.uuid4().hex[:12]}"
        calc_conf = confidence if confidence > 0.0 else max(probability, 1.0 - probability)

        try:
            cursor.execute("""
                INSERT INTO predictions_log (
                    prediction_id, machine_id, volt, rotate, pressure, vibration, 
                    prediction, probability, confidence, latency_ms, model_version,
                    risk_level, recommended_action
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (pred_id, machine_id, volt, rotate, pressure, vibration, 
                  prediction, probability, calc_conf, latency_ms, model_version,
                  risk_level, recommended_action))
            conn.commit()
            logger.info(f"Logged prediction to DB: ID={pred_id} | Machine={machine_id} | Fail={prediction}")
            return pred_id
        except Exception as e:
            logger.error(f"Failed to log prediction to SQLite: {e}")
            return pred_id
        finally:
            conn.close()

    def log_exception(self, endpoint: str, error_message: str, stack_trace: str):
        """Log a system exception to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO exceptions_log (endpoint, error_message, stack_trace)
                VALUES (?, ?, ?)
            """, (endpoint, error_message, stack_trace))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log exception to SQLite: {e}")
        finally:
            conn.close()

    def log_request(self, method: str, path: str, status_code: int, latency_ms: float):
        """Log an API HTTP request to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO api_requests_log (method, path, status_code, latency_ms)
                VALUES (?, ?, ?, ?)
            """, (method, path, status_code, latency_ms))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log API request to SQLite: {e}")
        finally:
            conn.close()

    def check_connection(self) -> bool:
        """Check if database is connected and responsive."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Database connection health check failed: {e}")
            return False

    def get_kpis(self):
        """Fetch logged prediction statistics for dashboard aggregation."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) as total, SUM(prediction) as failures FROM predictions_log")
            row = cursor.fetchone()
            total_predictions = row['total'] or 0
            failures = row['failures'] or 0
            return {
                "total_predictions": total_predictions,
                "failures_logged": failures,
            }
        except Exception as e:
            logger.error(f"Error fetching predictions log statistics: {e}")
            return {"total_predictions": 0, "failures_logged": 0}
        finally:
            conn.close()

    def get_filtered_predictions(self, search: Optional[str] = None, machine_id: Optional[int] = None,
                                 risk_level: Optional[str] = None, prediction: Optional[int] = None,
                                 limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Fetch predictions with search, filtering, and pagination support."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            query = "SELECT * FROM predictions_log WHERE 1=1"
            params: List[Any] = []

            if machine_id is not None:
                query += " AND machine_id = ?"
                params.append(machine_id)
            if risk_level and risk_level.lower() != "all":
                query += " AND LOWER(risk_level) = LOWER(?)"
                params.append(risk_level)
            if prediction is not None:
                query += " AND prediction = ?"
                params.append(prediction)
            if search:
                query += " AND (prediction_id LIKE ? OR CAST(machine_id AS TEXT) LIKE ? OR risk_level LIKE ?)"
                search_param = f"%{search}%"
                params.extend([search_param, search_param, search_param])

            # Count query for total matching rows
            count_query = f"SELECT COUNT(*) as cnt FROM ({query})"
            cursor.execute(count_query, params)
            total_rows = cursor.fetchone()['cnt']

            # Pagination query
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()
            logs = []
            for r in rows:
                d = dict(r)
                if not d.get("prediction_id"):
                    d["prediction_id"] = f"pred_legacy_{d['id']}"
                if d.get("confidence") is None:
                    prob = d.get("probability", 0.5)
                    d["confidence"] = round(max(prob, 1.0 - prob), 4)
                if not d.get("model_version"):
                    d["model_version"] = "v2.0.0"
                logs.append(d)

            return {
                "total": total_rows,
                "limit": limit,
                "offset": offset,
                "logs": logs,
            }
        except Exception as e:
            logger.error(f"Error reading filtered predictions log: {e}")
            return {"total": 0, "limit": limit, "offset": offset, "logs": []}
        finally:
            conn.close()

    def export_predictions_csv(self, search: Optional[str] = None, risk_level: Optional[str] = None) -> str:
        """Export predictions log to CSV string format."""
        result = self.get_filtered_predictions(search=search, risk_level=risk_level, limit=5000, offset=0)
        logs = result["logs"]

        output = io.StringIO()
        if not logs:
            writer = csv.writer(output)
            writer.writerow(["prediction_id", "timestamp", "machine_id", "volt", "rotate", "pressure", "vibration", "prediction", "probability", "confidence", "latency_ms", "model_version", "risk_level"])
            return output.getvalue()

        fieldnames = ["prediction_id", "timestamp", "machine_id", "volt", "rotate", "pressure", "vibration", "prediction", "probability", "confidence", "latency_ms", "model_version", "risk_level", "recommended_action"]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for log in logs:
            writer.writerow(log)

        return output.getvalue()

    def get_recent_predictions(self, limit: int = 10):
        """Fetch the most recent predictions logged in the database."""
        result = self.get_filtered_predictions(limit=limit)
        return result["logs"]
