"""
WebSocket endpoint for real-time workflow execution progress.

Provides live updates during workflow execution for:
- Node start/complete events
- Progress percentage
- Error notifications
- Final completion status

Usage:
    const ws = new WebSocket('/api/v1/workflows/executions/{execution_id}/progress');

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // data.type: 'node_start' | 'node_complete' | 'node_error' | 'progress' | 'complete' | 'error'
        // data.data: event-specific payload
        // data.timestamp: ISO timestamp
    };
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Set, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.supabase import supabase_service

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """
    Manage WebSocket connections for execution progress streaming.

    Supports multiple clients watching the same execution.
    """

    def __init__(self):
        # execution_id -> set of connected websockets
        self.connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, execution_id: str, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()

        if execution_id not in self.connections:
            self.connections[execution_id] = set()

        self.connections[execution_id].add(websocket)
        logger.info(f"WS connected: execution={execution_id[:8]}...")

    def disconnect(self, execution_id: str, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if execution_id in self.connections:
            self.connections[execution_id].discard(websocket)
            if not self.connections[execution_id]:
                del self.connections[execution_id]

        logger.info(f"WS disconnected: execution={execution_id[:8]}...")

    async def broadcast(self, execution_id: str, event_type: str, data: dict):
        """
        Broadcast a message to all clients watching an execution.

        Args:
            execution_id: The execution to broadcast to
            event_type: Event type (node_start, node_complete, progress, etc.)
            data: Event-specific payload
        """
        if execution_id not in self.connections:
            return

        message = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        message_json = json.dumps(message)

        dead_connections = set()
        for websocket in self.connections[execution_id]:
            try:
                await websocket.send_text(message_json)
            except Exception:
                dead_connections.add(websocket)

        # Clean up dead connections
        for ws in dead_connections:
            self.connections[execution_id].discard(ws)

    def has_connections(self, execution_id: str) -> bool:
        """Check if any clients are watching an execution."""
        return execution_id in self.connections and len(self.connections[execution_id]) > 0


# Singleton connection manager
manager = ConnectionManager()


@router.websocket("/executions/{execution_id}/progress")
async def execution_progress(websocket: WebSocket, execution_id: str):
    """
    WebSocket endpoint for real-time execution progress.

    Sends events:
    - initial_state: Current execution state on connect
    - node_start: {node_id, node_type, timestamp}
    - node_complete: {node_id, duration_ms, outputs_summary}
    - node_error: {node_id, error}
    - progress: {percent, current_node, completed_nodes, total_nodes}
    - complete: {status, duration_ms, outputs}
    - error: {message}
    - heartbeat: Periodic keepalive
    """
    await manager.connect(execution_id, websocket)

    try:
        # Send current state on connect
        execution = supabase_service.client.table("wf_executions").select(
            "status, progress, output_data, error_message, duration_ms"
        ).eq("id", execution_id).single().execute()

        if execution.data:
            await websocket.send_json({
                "type": "initial_state",
                "data": execution.data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # If already complete, send completion event and close
            if execution.data["status"] in ("completed", "failed", "cancelled"):
                await websocket.send_json({
                    "type": "complete",
                    "data": {
                        "status": execution.data["status"],
                        "duration_ms": execution.data.get("duration_ms"),
                        "outputs": execution.data.get("output_data"),
                        "error": execution.data.get("error_message"),
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return
        else:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "Execution not found"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return

        # Keep connection alive and listen for ping/commands
        while True:
            try:
                # Wait for client messages (ping/pong or commands)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30,
                )

                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "status":
                    # Client requesting current status
                    current = supabase_service.client.table("wf_executions").select(
                        "status, progress"
                    ).eq("id", execution_id).single().execute()
                    if current.data:
                        await websocket.send_json({
                            "type": "status_update",
                            "data": current.data,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                # Check if execution completed while we were waiting
                current = supabase_service.client.table("wf_executions").select(
                    "status, output_data, error_message, duration_ms"
                ).eq("id", execution_id).single().execute()

                if current.data and current.data["status"] in ("completed", "failed", "cancelled"):
                    await websocket.send_json({
                        "type": "complete",
                        "data": {
                            "status": current.data["status"],
                            "duration_ms": current.data.get("duration_ms"),
                            "outputs": current.data.get("output_data"),
                            "error": current.data.get("error_message"),
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    break

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(execution_id, websocket)


# Helper functions for broadcasting from engine
async def broadcast_node_start(execution_id: str, node_id: str, node_type: str, node_label: str = None):
    """Broadcast node start event."""
    await manager.broadcast(execution_id, "node_start", {
        "node_id": node_id,
        "node_type": node_type,
        "node_label": node_label or node_type,
    })


async def broadcast_node_complete(execution_id: str, node_id: str, duration_ms: float, outputs_summary: dict = None):
    """Broadcast node complete event."""
    await manager.broadcast(execution_id, "node_complete", {
        "node_id": node_id,
        "duration_ms": duration_ms,
        "outputs_summary": outputs_summary or {},
    })


async def broadcast_node_error(execution_id: str, node_id: str, error: str):
    """Broadcast node error event."""
    await manager.broadcast(execution_id, "node_error", {
        "node_id": node_id,
        "error": error,
    })


async def broadcast_progress(
    execution_id: str,
    percent: int,
    current_node: Optional[str],
    completed_nodes: list,
    total_nodes: int,
):
    """Broadcast progress update."""
    await manager.broadcast(execution_id, "progress", {
        "percent": percent,
        "current_node": current_node,
        "completed_nodes": completed_nodes,
        "total_nodes": total_nodes,
    })


async def broadcast_complete(execution_id: str, status: str, duration_ms: int, outputs: dict = None, error: str = None):
    """Broadcast execution complete event."""
    await manager.broadcast(execution_id, "complete", {
        "status": status,
        "duration_ms": duration_ms,
        "outputs": outputs,
        "error": error,
    })


# Export for use in engine
__all__ = [
    "router",
    "manager",
    "broadcast_node_start",
    "broadcast_node_complete",
    "broadcast_node_error",
    "broadcast_progress",
    "broadcast_complete",
]
