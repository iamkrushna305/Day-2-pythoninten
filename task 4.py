
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ============================================================
# Configuration
# ============================================================

SERVICE_NAME = "FDE AI Service"
SERVICE_VERSION = "1.0.0"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
APP_ENV = os.getenv("APP_ENV", "development").lower()


def get_secret(name: str) -> str | None:
    """Read a Docker secret first, then an environment variable."""
    secret_file = Path("/run/secrets") / name.lower()

    try:
        if secret_file.is_file():
            value = secret_file.read_text(encoding="utf-8").strip()
            if value:
                return value
    except OSError:
        pass

    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
API_AUTH_TOKEN = get_secret("API_AUTH_TOKEN")

app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="FDE AI Service with inventory, CRM, approvals and monitoring.",
)


# ============================================================
# Prometheus monitoring
# ============================================================

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

HTTP_ERRORS = Counter(
    "http_errors_total",
    "Total HTTP error responses",
    ["method", "endpoint", "status"],
)

HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

LLM_CALLS = Counter(
    "llm_calls_total",
    "Total LLM calls",
    ["model", "status"],
)

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total tokens used by the LLM",
    ["model", "type"],
)

PENDING_APPROVALS = Gauge(
    "agent_pending_approvals",
    "Number of pending agent approvals",
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        # Avoid high-cardinality paths such as individual ticket IDs.
        path = request.url.path
        if path.startswith("/approvals/"):
            path = "/approvals/{approval_id}"
        elif path.startswith("/api/inventory/"):
            path = "/api/inventory/{sku}"

        method = request.method
        status = str(status_code)

        HTTP_REQUESTS.labels(method, path, status).inc()
        HTTP_DURATION.labels(method, path).observe(
            time.perf_counter() - start
        )

        if status_code >= 400:
            HTTP_ERRORS.labels(method, path, status).inc()


# ============================================================
# Authentication
# ============================================================

def require_auth(
    authorization: str | None = Header(default=None),
) -> None:
    # Fail closed in production if no API token is configured.
    if not API_AUTH_TOKEN:
        if APP_ENV == "production":
            raise HTTPException(
                status_code=503,
                detail="API authentication is not configured.",
            )
        return

    expected = f"Bearer {API_AUTH_TOKEN}"

    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================
# Demo data - in memory only
# Replace these dictionaries with a database for production.
# ============================================================

inventory: dict[str, dict[str, Any]] = {
    "SKU-1001": {
        "sku": "SKU-1001",
        "name": "Laptop",
        "quantity": 25,
        "price": 65000.0,
    },
    "SKU-1002": {
        "sku": "SKU-1002",
        "name": "Wireless Mouse",
        "quantity": 100,
        "price": 750.0,
    },
    "SKU-1003": {
        "sku": "SKU-1003",
        "name": "Keyboard",
        "quantity": 50,
        "price": 1200.0,
    },
}

leads: dict[str, dict[str, Any]] = {
    "LEAD-001": {
        "lead_id": "LEAD-001",
        "name": "Sample Customer",
        "status": "new",
        "notes": "",
    }
}

tickets: dict[str, dict[str, Any]] = {}
approvals: dict[str, dict[str, Any]] = {}
audit_log: list[dict[str, Any]] = []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_audit(
    action: str,
    details: dict[str, Any],
    actor: str = "system",
) -> None:
    audit_log.append(
        {
            "event_id": str(uuid.uuid4()),
            "timestamp": utc_now(),
            "actor": actor,
            "action": action,
            "details": details,
        }
    )


def update_pending_gauge() -> None:
    PENDING_APPROVALS.set(
        sum(
            1
            for item in approvals.values()
            if item["status"] == "pending"
        )
    )


# ============================================================
# Request models
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "reject"]


# ============================================================
# Root and health endpoints
# ============================================================

@app.get("/", response_class=JSONResponse)
def home():
    return {
        "service": "FDE AI Service",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health/live")
def health_live():
    return {
        "service": SERVICE_NAME,
        "status": "healthy",
    }


@app.get("/health/ready")
def health_ready():
    # The service can start without an OpenAI key, but the AI
    # endpoint will not work until a key is configured.
    return {
        "service": SERVICE_NAME,
        "status": "ready",
        "ai_configured": bool(OPENAI_API_KEY),
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================
# Inventory API
# ============================================================

@app.get("/api/inventory", dependencies=[Depends(require_auth)])
def list_inventory():
    return {"items": list(inventory.values())}


@app.get(
    "/api/inventory/{sku}",
    dependencies=[Depends(require_auth)],
)
def get_inventory_item(sku: str):
    item = inventory.get(sku.upper())

    if item is None:
        raise HTTPException(
            status_code=404,
            detail="Inventory item not found.",
        )

    record_audit("inventory_lookup", {"sku": sku.upper()})
    return item


# ============================================================
# CRM API
# ============================================================

@app.get("/api/leads", dependencies=[Depends(require_auth)])
def list_leads():
    return {"leads": list(leads.values())}


@app.get(
    "/api/leads/{lead_id}",
    dependencies=[Depends(require_auth)],
)
def get_lead(lead_id: str):
    lead = leads.get(lead_id.upper())

    if lead is None:
        raise HTTPException(status_code=404, detail="Lead not found.")

    return lead


# ============================================================
# Ticket API
# ============================================================

@app.get("/api/tickets", dependencies=[Depends(require_auth)])
def list_tickets():
    return {"tickets": list(tickets.values())}


# ============================================================
# Approval workflow
# ============================================================

def create_approval(
    action: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    approval_id = str(uuid.uuid4())

    approval = {
        "approval_id": approval_id,
        "action": action,
        "payload": payload,
        "status": "pending",
        "created_at": utc_now(),
        "decided_at": None,
        "result": None,
    }

    approvals[approval_id] = approval
    update_pending_gauge()

    record_audit(
        "approval_requested",
        {"approval_id": approval_id, "action": action},
    )

    return approval


def execute_approved_action(
    action: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Execute an action only after explicit approval."""

    if action == "lead_update":
        lead_id = payload["lead_id"].upper()
        lead = leads.get(lead_id)

        if lead is None:
            raise ValueError(f"Lead {lead_id} does not exist.")

        lead["status"] = payload["status"]
        lead["notes"] = payload.get("notes", lead["notes"])

        record_audit(
            "lead_updated",
            {"lead_id": lead_id, "status": lead["status"]},
            actor="approved_agent",
        )

        return {"lead": lead}

    if action == "ticket_create":
        ticket_id = f"TICKET-{uuid.uuid4().hex[:8].upper()}"

        ticket = {
            "ticket_id": ticket_id,
            "title": payload["title"],
            "description": payload["description"],
            "priority": payload["priority"],
            "status": "open",
            "created_at": utc_now(),
        }

        tickets[ticket_id] = ticket

        record_audit(
            "ticket_created",
            {"ticket_id": ticket_id},
            actor="approved_agent",
        )

        return {"ticket": ticket}

    raise ValueError("Unsupported approval action.")


@app.get("/approvals", dependencies=[Depends(require_auth)])
def list_approvals():
    return {"approvals": list(approvals.values())}


@app.post(
    "/approvals/{approval_id}/decision",
    dependencies=[Depends(require_auth)],
)
def decide_approval(
    approval_id: str,
    request: ApprovalDecision,
):
    approval = approvals.get(approval_id)

    if approval is None:
        raise HTTPException(
            status_code=404,
            detail="Approval request not found.",
        )

    if approval["status"] != "pending":
        raise HTTPException(
            status_code=409,
            detail="This approval has already been decided.",
        )

    approval["decided_at"] = utc_now()

    if request.decision == "reject":
        approval["status"] = "rejected"
        approval["result"] = {"message": "Action rejected."}

        record_audit(
            "approval_rejected",
            {"approval_id": approval_id},
        )

        update_pending_gauge()
        return approval

    try:
        result = execute_approved_action(
            approval["action"],
            approval["payload"],
        )
        approval["status"] = "approved"
        approval["result"] = result

        record_audit(
            "approval_accepted",
            {"approval_id": approval_id},
        )

    except (ValueError, KeyError) as exc:
        approval["status"] = "failed"
        approval["result"] = {"message": str(exc)}

        record_audit(
            "approved_action_failed",
            {"approval_id": approval_id},
        )

        update_pending_gauge()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    update_pending_gauge()
    return approval


# ============================================================
# Agent tools
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inventory_lookup",
            "description": "Look up a product using its SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Product SKU, e.g. SKU-1001",
                    }
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_lead_update",
            "description": (
                "Request approval to update a CRM lead. "
                "This does not update the lead immediately."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lead_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": [
                            "new",
                            "contacted",
                            "qualified",
                            "won",
                            "lost",
                        ],
                    },
                    "notes": {"type": "string"},
                },
                "required": ["lead_id", "status", "notes"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_ticket_create",
            "description": (
                "Request approval to create a support ticket. "
                "The ticket is created only after approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                    },
                },
                "required": ["title", "description", "priority"],
                "additionalProperties": False,
            },
        },
    },
]


def dispatch_tool(
    name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Allow only explicitly registered tools."""

    if name == "inventory_lookup":
        sku = str(args["sku"]).upper()
        item = inventory.get(sku)

        record_audit("agent_inventory_lookup", {"sku": sku})

        if item is None:
            return {"error": f"Product {sku} was not found."}

        return {"item": item}

    if name == "request_lead_update":
        lead_id = str(args["lead_id"]).upper()

        if lead_id not in leads:
            return {"error": f"Lead {lead_id} was not found."}

        approval = create_approval(
            "lead_update",
            {
                "lead_id": lead_id,
                "status": args["status"],
                "notes": args["notes"],
            },
        )

        return {
            "message": "Approval required before updating the lead.",
            "approval_id": approval["approval_id"],
            "status": approval["status"],
        }

    if name == "request_ticket_create":
        approval = create_approval(
            "ticket_create",
            {
                "title": args["title"],
                "description": args["description"],
                "priority": args["priority"],
            },
        )

        return {
            "message": "Approval required before creating the ticket.",
            "approval_id": approval["approval_id"],
            "status": approval["status"],
        }

    return {"error": "Tool is not allowed."}


# ============================================================
# OpenAI agent API
# ============================================================

@app.post("/agent/chat", dependencies=[Depends(require_auth)])
def agent_chat(request: ChatRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenAI API key is not configured. "
                "Set the OPENAI_API_KEY environment variable."
            ),
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are the FDE AI Service assistant. "
                "Use inventory_lookup to answer inventory questions. "
                "For CRM updates and ticket creation, request approval "
                "using the available tools. Never claim an action was "
                "completed when it is only pending approval. "
                "Be clear and concise."
            ),
        },
        {"role": "user", "content": request.message},
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0

    try:
        # Limit tool-call rounds to avoid an endless agent loop.
        for _ in range(5):
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            LLM_CALLS.labels(OPENAI_MODEL, "success").inc()

            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += (
                    response.usage.completion_tokens
                )

            choice = response.choices[0]
            assistant_message = choice.message

            messages.append(
                assistant_message.model_dump(exclude_none=True)
            )

            if not assistant_message.tool_calls:
                LLM_TOKENS.labels(
                    OPENAI_MODEL, "prompt"
                ).inc(total_prompt_tokens)
                LLM_TOKENS.labels(
                    OPENAI_MODEL, "completion"
                ).inc(total_completion_tokens)

                record_audit(
                    "agent_chat",
                    {"model": OPENAI_MODEL},
                )

                return {
                    "service": SERVICE_NAME,
                    "response": assistant_message.content or "",
                    "model": OPENAI_MODEL,
                    "usage": {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                    },
                }

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name

                try:
                    arguments = json.loads(
                        tool_call.function.arguments or "{}"
                    )
                    result = dispatch_tool(tool_name, arguments)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    result = {"error": "Invalid tool arguments."}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

        raise HTTPException(
            status_code=502,
            detail="Agent reached its maximum tool-call limit.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        LLM_CALLS.labels(OPENAI_MODEL, "error").inc()

        # Avoid returning API keys or internal exception details.
        record_audit(
            "agent_error",
            {"error_type": type(exc).__name__},
        )

        raise HTTPException(
            status_code=502,
            detail="The AI request failed. Check server logs and API configuration.",
        ) from exc


# ============================================================
# Audit API
# ============================================================

@app.get("/api/audit", dependencies=[Depends(require_auth)])
def get_audit_log(limit: int = 100):
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=422,
            detail="limit must be between 1 and 1000.",
        )

    return {"events": audit_log[-limit:]}


# ============================================================
# Run directly with: python "task 4.py"
# Recommended: python -m uvicorn "task 4:app" --reload
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "task 4:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
