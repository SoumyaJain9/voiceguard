import os
import json
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from .config import settings

# Attempt Supabase initialization if credentials exist
supabase_client = None
if settings.SUPABASE_URL and settings.SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        print("[+] [Database] Supabase Cloud Logging Initialized Successfully")
    except Exception as e:
        print(f"[*] [Database] Supabase initialization skipped or failed: {e}")

class AuditDatabase:
    """
    Manages audit logging for deepfake detection requests.
    Supports dual-logging: Supabase (Cloud) + SQLite (Local fallback for offline demos).
    """
    def __init__(self, db_path: str = settings.SQLITE_DB_PATH):
        self.db_path = db_path
        self._init_sqlite()

    def _init_sqlite(self):
        """Initializes the local SQLite database schema."""
        try:
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS scan_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        ip_address TEXT,
                        language TEXT,
                        classification TEXT,
                        confidence REAL,
                        latency_seconds REAL,
                        status TEXT,
                        explanation TEXT,
                        request_meta TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"[*] [Database] Local SQLite init warning: {e}")

    def log_inference_event(self, event_data: Dict[str, Any]):
        """Logs the event asynchronously to avoid blocking the API response."""
        # 1. Local SQLite Log
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO scan_logs (
                        timestamp, ip_address, language, classification, 
                        confidence, latency_seconds, status, explanation, request_meta
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_data.get("created_at", datetime.now(timezone.utc).isoformat()),
                    event_data.get("ip_address", "127.0.0.1"),
                    event_data.get("input_language", "Unknown"),
                    event_data.get("result_classification", "UNKNOWN"),
                    event_data.get("result_confidence", 0.0),
                    event_data.get("latency_seconds", 0.0),
                    event_data.get("status", "success"),
                    event_data.get("response_json", {}).get("explanation", ""),
                    json.dumps(event_data.get("request_json", {}))
                ))
                conn.commit()
        except Exception as e:
            print(f"[-] [Database] Local SQLite log error: {e}")

        # 2. Supabase Cloud Log (if available)
        if supabase_client:
            try:
                supabase_client.table("api_logs").insert(event_data).execute()
            except Exception as e:
                print(f"[-] [Database] Supabase cloud log error: {e}")

    def get_recent_scans(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves recent audit logs for the web dashboard."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, ip_address, language, classification, 
                           confidence, latency_seconds, status, explanation
                    FROM scan_logs
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            print(f"[-] [Database] Failed to fetch scan logs: {e}")
            return []

# Global audit manager instance
audit_db = AuditDatabase()

def log_event_in_background(data: Dict[str, Any]):
    """Background execution wrapper for logging."""
    audit_db.log_inference_event(data)
