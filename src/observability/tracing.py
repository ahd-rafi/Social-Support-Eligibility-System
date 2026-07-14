"""Langfuse v2 tracing integration for observability."""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from src.config import settings

logger = logging.getLogger(__name__)

# Try to import Langfuse (optional dependency)
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse not installed. Tracing disabled.")


class TracingClient:
    """Langfuse tracing client wrapper."""

    def __init__(self):
        """Initialize Langfuse client."""
        self.client: Optional[Langfuse] = None
        self.enabled = False
        
        if LANGFUSE_AVAILABLE and settings.langfuse_public_key and settings.langfuse_secret_key:
            # Check if credentials are placeholders
            if "placeholder" in settings.langfuse_secret_key.lower():
                logger.info("Langfuse tracing disabled (placeholder credentials)")
                self.enabled = False
            else:
                try:
                    self.client = Langfuse(
                        public_key=settings.langfuse_public_key,
                        secret_key=settings.langfuse_secret_key,
                        host=settings.langfuse_host,
                    )
                    self.enabled = True
                    logger.info(f"Langfuse tracing enabled: {settings.langfuse_host}")
                except Exception as e:
                    logger.warning(f"Langfuse init failed (non-critical): {str(e)[:100]}")
                    self.enabled = False
        else:
            logger.info("Langfuse tracing disabled (no credentials)")

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a new trace.
        
        Args:
            name: Trace name
            user_id: User ID
            session_id: Session ID
            metadata: Additional metadata
            
        Returns:
            Trace ID if successful, None otherwise
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            trace = self.client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
            return trace.id
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None

    def log_event(
        self,
        trace_id: str,
        name: str,
        level: str = "DEFAULT",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log an event within a trace.
        
        Args:
            trace_id: Trace ID
            name: Event name
            level: Event level (DEFAULT, DEBUG, WARNING, ERROR)
            metadata: Event metadata
        """
        if not self.enabled or not self.client:
            return
        
        try:
            self.client.event(
                trace_id=trace_id,
                name=name,
                level=level,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def log_span(
        self,
        trace_id: str,
        name: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: str = "DEFAULT",
    ):
        """Log a span (operation duration) within a trace.
        
        Args:
            trace_id: Trace ID
            name: Span name
            start_time: Start timestamp
            end_time: End timestamp
            metadata: Span metadata
            level: Log level
        """
        if not self.enabled or not self.client:
            return
        
        try:
            self.client.span(
                trace_id=trace_id,
                name=name,
                start_time=start_time,
                end_time=end_time or datetime.now(),
                metadata=metadata or {},
                level=level,
            )
        except Exception as e:
            logger.error(f"Failed to log span: {e}")

    def log_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        metadata: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        """Log an LLM generation within a trace.
        
        Args:
            trace_id: Trace ID
            name: Generation name
            model: Model name
            prompt: Input prompt
            completion: Output completion
            metadata: Additional metadata
            start_time: Start timestamp
            end_time: End timestamp
        """
        if not self.enabled or not self.client:
            return
        
        try:
            self.client.generation(
                trace_id=trace_id,
                name=name,
                model=model,
                prompt=prompt,
                completion=completion,
                metadata=metadata or {},
                start_time=start_time or datetime.now(),
                end_time=end_time or datetime.now(),
            )
        except Exception as e:
            logger.error(f"Failed to log generation: {e}")

    def flush(self):
        """Flush any pending traces to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                # Suppress non-critical flush errors (placeholder credentials, network issues, etc.)
                logger.debug(f"Langfuse flush skipped: {str(e)[:100]}")


# Global tracing client
_tracing_client: Optional[TracingClient] = None


def get_tracing_client() -> TracingClient:
    """Get or create global tracing client.
    
    Returns:
        TracingClient instance
    """
    global _tracing_client
    if _tracing_client is None:
        _tracing_client = TracingClient()
    return _tracing_client


# Convenience functions for agent tracing

def trace_agent_start(
    application_id: str,
    agent_name: str,
    input_state: Dict[str, Any],
) -> Optional[str]:
    """Trace the start of an agent execution.
    
    Args:
        application_id: Application ID
        agent_name: Agent name
        input_state: Input state
        
    Returns:
        Trace ID if tracing enabled
    """
    client = get_tracing_client()
    
    return client.create_trace(
        name=f"agent_{agent_name}",
        session_id=application_id,
        metadata={
            "agent": agent_name,
            "application_id": application_id,
            "input_state_keys": list(input_state.keys()),
        },
    )


def trace_agent_end(
    trace_id: Optional[str],
    agent_name: str,
    output_state: Dict[str, Any],
    success: bool = True,
):
    """Trace the end of an agent execution.
    
    Args:
        trace_id: Trace ID
        agent_name: Agent name
        output_state: Output state
        success: Whether execution succeeded
    """
    if not trace_id:
        return
    
    client = get_tracing_client()
    
    client.log_event(
        trace_id=trace_id,
        name=f"agent_{agent_name}_complete",
        level="DEFAULT" if success else "ERROR",
        metadata={
            "agent": agent_name,
            "success": success,
            "output_state_keys": list(output_state.keys()),
        },
    )


def trace_llm_call(
    trace_id: Optional[str],
    model: str,
    prompt: str,
    completion: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Trace an LLM call.
    
    Args:
        trace_id: Trace ID
        model: Model name
        prompt: Input prompt
        completion: Output completion
        metadata: Additional metadata
    """
    if not trace_id:
        return
    
    client = get_tracing_client()
    
    client.log_generation(
        trace_id=trace_id,
        name="llm_generation",
        model=model,
        prompt=prompt,
        completion=completion,
        metadata=metadata or {},
    )


def trace_classifier_prediction(
    trace_id: Optional[str],
    features: Dict[str, Any],
    prediction: Dict[str, Any],
):
    """Trace a classifier prediction.
    
    Args:
        trace_id: Trace ID
        features: Input features
        prediction: Prediction result
    """
    if not trace_id:
        return
    
    client = get_tracing_client()
    
    client.log_event(
        trace_id=trace_id,
        name="classifier_prediction",
        level="DEFAULT",
        metadata={
            "features": features,
            "prediction": prediction,
        },
    )


def trace_validation_issue(
    trace_id: Optional[str],
    issue: Dict[str, Any],
):
    """Trace a validation issue.
    
    Args:
        trace_id: Trace ID
        issue: Validation issue details
    """
    if not trace_id:
        return
    
    client = get_tracing_client()
    
    level = "WARNING" if issue.get("severity") == "CRITICAL" else "DEFAULT"
    
    client.log_event(
        trace_id=trace_id,
        name="validation_issue",
        level=level,
        metadata=issue,
    )


if __name__ == "__main__":
    # Test tracing
    logging.basicConfig(level=logging.INFO)
    
    client = get_tracing_client()
    
    if client.enabled:
        print("✓ Langfuse tracing enabled")
        
        # Test trace
        trace_id = client.create_trace(
            name="test_trace",
            user_id="test_user",
            metadata={"test": True},
        )
        
        if trace_id:
            print(f"✓ Created trace: {trace_id}")
            
            client.log_event(
                trace_id=trace_id,
                name="test_event",
                metadata={"message": "Hello from Langfuse!"},
            )
            
            client.flush()
            print("✓ Flushed traces")
    else:
        print("✗ Langfuse tracing disabled")
