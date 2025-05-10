"""
Monitoring and observability for the Shrek Trading Platform.
Provides OpenTelemetry instrumentation for distributed tracing, metrics, and logging.
"""

import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Metrics
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Logging
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

logger = logging.getLogger(__name__)

# Get configuration from environment variables
OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() == "true"
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "shrek-trading-platform")
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")


def setup_telemetry():
    """Initialize OpenTelemetry instrumentation."""
    if not OTEL_ENABLED:
        logger.info("OpenTelemetry instrumentation is disabled")
        return

    # Create a resource to identify the service
    resource = Resource.create({SERVICE_NAME: OTEL_SERVICE_NAME})

    # Set up tracing
    tracer_provider = TracerProvider(resource=resource)
    
    # Configure span processor for OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT)
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # Set the global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Set up metrics
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT)
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    
    # Set up logging
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=OTEL_EXPORTER_OTLP_ENDPOINT))
    )
    set_logger_provider(logger_provider)
    
    # Add OpenTelemetry logging handler to the root logger
    handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)
    
    # Auto-instrument libraries
    RequestsInstrumentor().instrument()
    RedisInstrumentor().instrument()
    
    logger.info("OpenTelemetry instrumentation initialized")


def instrument_fastapi(app):
    """Instrument a FastAPI application with OpenTelemetry."""
    if OTEL_ENABLED:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry")


def instrument_sqlalchemy(engine):
    """Instrument a SQLAlchemy engine with OpenTelemetry."""
    if OTEL_ENABLED:
        SQLAlchemyInstrumentor().instrument(engine=engine)
        logger.info("SQLAlchemy engine instrumented with OpenTelemetry")


def trace_method(name: Optional[str] = None):
    """
    Decorator to add OpenTelemetry tracing to a method.
    
    Args:
        name: Optional custom name for the span
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OTEL_ENABLED:
                return func(*args, **kwargs)
            
            # Get the tracer
            tracer = trace.get_tracer(OTEL_SERVICE_NAME)
            
            # Create a span
            span_name = name or f"{func.__module__}.{func.__name__}"
            with tracer.start_as_current_span(span_name) as span:
                # Add the function name as a span attribute
                span.set_attribute("function.name", func.__name__)
                
                # Add class name if it's a method
                if args and hasattr(args[0], "__class__"):
                    span.set_attribute("class.name", args[0].__class__.__name__)
                
                # Call the original function
                result = func(*args, **kwargs)
                return result
                
        return wrapper
    
    return decorator


def trace_async_method(name: Optional[str] = None):
    """
    Decorator to add OpenTelemetry tracing to an async method.
    
    Args:
        name: Optional custom name for the span
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not OTEL_ENABLED:
                return await func(*args, **kwargs)
            
            # Get the tracer
            tracer = trace.get_tracer(OTEL_SERVICE_NAME)
            
            # Create a span
            span_name = name or f"{func.__module__}.{func.__name__}"
            with tracer.start_as_current_span(span_name) as span:
                # Add the function name as a span attribute
                span.set_attribute("function.name", func.__name__)
                
                # Add class name if it's a method
                if args and hasattr(args[0], "__class__"):
                    span.set_attribute("class.name", args[0].__class__.__name__)
                
                # Call the original function
                result = await func(*args, **kwargs)
                return result
                
        return wrapper
    
    return decorator


class MetricsRecorder:
    """
    Helper class for recording OpenTelemetry metrics.
    """
    
    def __init__(self, meter_name: str = OTEL_SERVICE_NAME):
        """
        Initialize the metrics recorder.
        
        Args:
            meter_name: Name of the meter
        """
        self.meter = metrics.get_meter(meter_name)
        self._counters = {}
        self._gauges = {}
        self._histograms = {}
    
    def get_counter(self, name: str, description: str, unit: str = "1"):
        """
        Get or create a counter metric.
        
        Args:
            name: Name of the counter
            description: Description of the counter
            unit: Unit of the counter
            
        Returns:
            The counter
        """
        if name not in self._counters:
            self._counters[name] = self.meter.create_counter(
                name=name,
                description=description,
                unit=unit
            )
        return self._counters[name]
    
    def increment_counter(self, name: str, value: int = 1, attributes: Optional[Dict[str, str]] = None):
        """
        Increment a counter.
        
        Args:
            name: Name of the counter
            value: Value to increment by
            attributes: Optional attributes
        """
        if not OTEL_ENABLED:
            return
        
        counter = self.get_counter(name, f"Counter for {name}")
        counter.add(value, attributes or {})
    
    def get_gauge(self, name: str, description: str, unit: str = "1"):
        """
        Get or create a gauge metric.
        
        Args:
            name: Name of the gauge
            description: Description of the gauge
            unit: Unit of the gauge
            
        Returns:
            The gauge
        """
        if name not in self._gauges:
            self._gauges[name] = self.meter.create_up_down_counter(
                name=name,
                description=description,
                unit=unit
            )
        return self._gauges[name]
    
    def record_gauge(self, name: str, value: float, attributes: Optional[Dict[str, str]] = None):
        """
        Record a gauge value.
        
        Args:
            name: Name of the gauge
            value: Value to record
            attributes: Optional attributes
        """
        if not OTEL_ENABLED:
            return
        
        gauge = self.get_gauge(name, f"Gauge for {name}")
        gauge.add(value, attributes or {})
    
    def get_histogram(self, name: str, description: str, unit: str = "1"):
        """
        Get or create a histogram metric.
        
        Args:
            name: Name of the histogram
            description: Description of the histogram
            unit: Unit of the histogram
            
        Returns:
            The histogram
        """
        if name not in self._histograms:
            self._histograms[name] = self.meter.create_histogram(
                name=name,
                description=description,
                unit=unit
            )
        return self._histograms[name]
    
    def record_histogram(self, name: str, value: float, attributes: Optional[Dict[str, str]] = None):
        """
        Record a histogram value.
        
        Args:
            name: Name of the histogram
            value: Value to record
            attributes: Optional attributes
        """
        if not OTEL_ENABLED:
            return
        
        histogram = self.get_histogram(name, f"Histogram for {name}")
        histogram.record(value, attributes or {})
