"""
Meta Budget Sync API — FastAPI (ready to deploy)
================================================
Reads a CSV of desired budgets and updates Meta (Facebook)
Campaigns or Ad Sets via the Graph API.

Includes:
- Dry-run mode (preview, no writes)
- Batch updates (chunks of 50)
- CSV encoding fallbacks
- Debug route to test Meta object IDs
"""

from __future__ import annotations
import csv
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

# ------------------------- Setup -------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
GRAPH_VERSION = os.getenv("GRAPH_VERSION", "v20.0")
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"
ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID")
LOG_DIR = os.getenv("LOG_DIR", "logs")
DEFAULT_DRY_RUN = os.getenv("DEFAULT_DRY_RUN", "true").lower() == "true"

os.makedirs(LOG_DIR, exist_ok=True)

app = FastAPI(title="Meta Budget Sync API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------- Security -------------------------
def require_api_key(header_key: Optional[str]):
    if API_KEY:
        if not header_key or header_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ------------------------- Helpers -------------------------
CURRENCY_FRACTION = {"JPY": 0, "KRW": 0, "VND": 0, "BHD": 3, "KWD": 3, "JOD": 3, "OMR": 3, "TND": 3}

def get_account_currency() -> str:
    url = f"{GRAPH_BASE}/{AD_ACCOUNT_ID}"
    params = {"fields": "currency", "access_token": ACCESS_TOKEN}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("currency", "USD")

def to_minor_units(amount: float, currency: str) -> int:
    exp = CURRENCY_FRACTION.get(currency.upper(), 2)
    return int(round(amount * (10 ** exp)))

MONEY_WORDS = re.compile(r"\b(usd|eur|gbp|dollars?|euros?|pounds?)\b", re.I)
NON_NUMERIC = re.compile(r"[^0-9,.\-]")

def parse_money(value: str) -> float:
    s = str(value or "").strip()
    if not s:
        raise ValueError("Empty value")
    s = MONEY_WORDS.sub("", s)
    s = s.replace("$", "").replace("£", "").replace("€", "")
    s = NON_NUMERIC.sub("", s).strip()
    if not s:
        raise ValueError(f"Cannot parse money from '{value}'")
    if re.fullmatch(r"\d{1,3}(\.\d{3})*,\d{2}", s) or re.fullmatch(r"\d+,\d{2}", s):
        s = s.replace(".", "").replace(",", ".")
    else:
        if s.count(".") <= 1:
            s = s.replace(",", "")
        if s.count(".") > 1:
            parts = s.split(".")
            s = "".join(parts[:-1]) + "." + parts[-1]
    return float(s)

# ------------------------- Data -------------------------
@dataclass
class BudgetChange:
    level: str
    object_id: str
    budget_type: str
    new_budget_minor: int
    human_amount: float
    currency: str
    reason: str = ""

REQUIRED_COLS = ["level", "id", "budget_type", "new_budget"]

class RateLimitError(Exception): pass

@retry(wait=wait_exponential(multiplier=1, min=1, max=60),
       stop=stop_after_attempt(6),
       retry=retry_if_exception_type((requests.HTTPError, RateLimitError)))
def graph_batch(requests_list: List[Dict]) -> List[Dict]:
    payload = {"access_token": ACCESS_TOKEN, "batch": json.dumps(requests_list)}
    resp = requests.post(GRAPH_BASE, data=payload, timeout=60)
    if resp.status_code == 429:
        raise RateLimitError("HTTP 429 from Graph API (batch)")
    resp.raise_for_status()
    return resp.json()

def validate_row(row: Dict[str, str]) -> Optional[str]:
    for k in REQUIRED_COLS:
        if not str(row.get(k, "")).strip():
            return f"Missing required column: {k}"
    if str(row["level"]).lower() not in {"adset", "campaign"}:
        return "level must be 'adset' or 'campaign'"
    if str(row["budget_type"]).lower() not in {"daily", "lifetime"}:
        return "budget_type must be 'daily' or 'lifetime'"
    return None

def plan_changes(rows: List[Dict[str, str]], default_currency: str) -> Tuple[List[BudgetChange], List[str]]:
    changes, errors = [], []
    for i, row in enumerate(rows, start=2):
        err = validate_row(row)
        if err:
            errors.append(f"Row {i}: {err}")
            continue
        try:
            human = parse_money(row["new_budget"])
        except Exception as e:
            errors.append(f"Row {i}: cannot parse new_budget '{row.get('new_budget')}' — {e}")
            continue
        cur = (row.get("currency") or default_currency).upper()
        changes.append(BudgetChange(
            row["level"].lower(), row["id"].strip(),
            row["budget_type"].lower(), to_minor_units(human, cur),
            human, cur, row.get("reason", "")
        ))
    return changes, errors

def build_payload(c: BudgetChange) -> Dict[str, str]:
    field = "daily_budget" if c.budget_type == "daily" else "lifetime_budget"
    return {field: str(c.new_budget_minor)}

def chunked(seq: List, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def apply_batch(changes: List[BudgetChange], dry: bool) -> List[Dict]:
    out = []
    for group in chunked(changes, 50):
        batch = []
        for c in group:
            body = "&".join(f"{k}={v}" for k, v in build_payload(c).items())
            batch.append({"method": "POST", "relative_url": f"/{c.object_id}", "body": body})
        out.extend([{"code": 200, "body": json.dumps({"success": True, "dry_run": True})} for _ in batch] if dry else graph_batch(batch))
    return out

def write_log(changes: List[BudgetChange], responses: List[Dict], cur: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"change_log_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_utc","level","id","budget_type","new_budget_major","currency","reason","status_code","response"])
        for c, r in zip(changes, responses):
            w.writerow([ts,c.level,c.object_id,c.budget_type,f"{c.human_amount:.2f}",c.currency or cur,c.reason,r.get("code"),r.get("body")])
    return path

# ------------------------- Models -------------------------
class SyncResult(BaseModel):
    successes: int
    failures: int
    total: int
    log_path: Optional[str]
    sample_errors: Optional[List[str]]

# ------------------------- Routes -------------------------
@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.post("/sync", response_model=SyncResult)
def sync(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    file: Optional[UploadFile] = File(None),
    csv_url: Optional[str] = Form(None),
    dry_run: Optional[bool] = Form(None),
):
    require_api_key(x_api_key)
    if not ACCESS_TOKEN or not AD_ACCOUNT_ID:
        raise HTTPException(status_code=500, detail="Missing Meta credentials")

    # Get file content
    if file:
        csv_bytes = file.file.read()
    elif csv_url:
        r = requests.get(csv_url, timeout=60)
        r.raise_for_status()
        csv_bytes = r.content
    else:
        raise HTTPException(status_code=400, detail="Provide a CSV file upload or csv_url")

    # Read CSV with encoding fallbacks
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, encoding="utf-8").fillna("")
    except UnicodeDecodeError:
        for enc in ("utf-8-sig", "cp1252", "latin1"):
            try:
                df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, encoding=enc).fillna("")
                break
            except Exception:
                continue
        else:
            raise HTTPException(status_code=400, detail="CSV parse error: encoding unsupported")

    rows = df.to_dict(orient="records")

    cur = get_account_currency()
    eff_dry = DEFAULT_DRY_RUN if dry_run is None else bool(dry_run)
    changes, errs = plan_changes(rows, cur)
    if not changes and errs:
        raise HTTPException(status_code=400, detail={"validation_errors": errs})

    try:
        res = apply_batch(changes, eff_dry)
    except RetryError:
        raise HTTPException(status_code=429, detail="Graph API retry exhausted")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Graph API error: {e}")

    ok = sum(1 for r in res if int(r.get("code", 200)) in (200, 201))
    log_path = write_log(changes, res, cur)
    sample = [str(r.get("body")) for r in res if int(r.get("code", 200)) >= 400][:5]

    return SyncResult(successes=ok, failures=len(res)-ok, total=len(res), log_path=log_path, sample_errors=sample or None)

@app.get("/debug_id")
def debug_id(id: str, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Type-aware debug: first fetch minimal data with metadata=1 to learn the node type,
    then fetch a safe field set for that type (Campaign vs AdSet).
    """
    require_api_key(x_api_key)

    # Step 1: minimal fetch + metadata to detect type
    meta_params = {
        "fields": "id,name",           # ultra-safe fields on both
        "metadata": "1",               # ask Graph API to include object type
        "access_token": ACCESS_TOKEN,
    }
    meta_resp = requests.get(f"{GRAPH_BASE}/{id}", params=meta_params, timeout=30)
    meta_json = meta_resp.json()
    node_type = None
    # Graph returns metadata like: {"metadata":{"type":"AdSet"}} when metadata=1
    try:
        node_type = (meta_json.get("metadata") or {}).get("type")
    except Exception:
        node_type = None

    # If the first call already errored, return it directly
    if meta_resp.status_code >= 400:
        return {"http": meta_resp.status_code, "data": meta_json, "node_type": node_type}

    # Step 2: choose fields based on detected type
    if node_type == "AdSet":
        fields = ",".join([
            "id","name","effective_status","status","configured_status",
            "daily_budget","lifetime_budget","campaign_id","start_time","end_time"
        ])
    elif node_type == "Campaign":
        # budget fields appear only if CBO/Advantage Campaign Budget is enabled
        fields = ",".join([
            "id","name","effective_status","status","configured_status",
            "buying_type","daily_budget","lifetime_budget","start_time","stop_time"
        ])
    else:
        # Unknown type: just return the minimal result we got
        return {"http": meta_resp.status_code, "data": meta_json, "node_type": node_type}

    # Step 3: type-specific fetch
    resp = requests.get(
        f"{GRAPH_BASE}/{id}",
        params={"fields": fields, "access_token": ACCESS_TOKEN},
        timeout=30,
    )
    return {"http": resp.status_code, "data": resp.json(), "node_type": node_type}
