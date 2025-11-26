"""OpenTelemetry tracing setup for the agent system."""

from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.sdk.resources import Resource


def setup_tracing(service_name: str = "agent-recommendation", enable: bool = True) -> Optional[trace.Tracer]:
    """
    Set up OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service
        enable: Whether to enable tracing
        
    Returns:
        Tracer instance or None if disabled
    """
    if not enable:
        return None
    
    # Create resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0"
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    
    # Add console exporter (in production, use OTLP exporter)
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    provider.add_span_processor(span_processor)
    
    return trace.get_tracer(service_name)


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a tracer instance for a specific component.
    
    Args:
        name: Component name
        
    Returns:
        Tracer instance
    """
    return trace.get_tracer(f"agent_recommendation.{name}")




