# advanced_handler.py
import runpod
import cloudpickle
import base64
import traceback
import logging
import time
import torch
import psutil
import gc
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExecutionStats:
    """Statistics about task execution"""
    start_time: float
    end_time: float
    gpu_memory_used: Optional[int] = None
    cpu_memory_used: Optional[int] = None
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

class ResourceMonitor:
    """Monitor system resources during execution"""
    @staticmethod
    def get_gpu_memory_used():
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
        return None

    @staticmethod
    def get_cpu_memory_used():
        return psutil.Process().memory_info().rss

    @staticmethod
    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

@contextmanager
def task_execution_context(task_id: str):
    """Context manager for task execution with resource monitoring"""
    stats = ExecutionStats(start_time=time.time(), end_time=0)
    
    try:
        # Clear GPU memory before execution
        ResourceMonitor.clear_gpu_memory()
        logger.info(f"Starting task {task_id}")
        
        yield stats
        
    finally:
        stats.end_time = time.time()
        stats.gpu_memory_used = ResourceMonitor.get_gpu_memory_used()
        stats.cpu_memory_used = ResourceMonitor.get_cpu_memory_used()
        
        # Log execution statistics
        logger.info(
            f"Task {task_id} completed in {stats.duration:.2f}s. "
            f"GPU Memory: {stats.gpu_memory_used or 'N/A'}, "
            f"CPU Memory: {stats.cpu_memory_used}"
        )
        
        # Clear resources
        ResourceMonitor.clear_gpu_memory()

class PayloadProcessor:
    """Handle payload serialization/deserialization"""
    @staticmethod
    def deserialize_payload(payload: Dict) -> tuple:
        try:
            func = cloudpickle.loads(
                base64.b64decode(payload['function'])
            )
            args = cloudpickle.loads(
                base64.b64decode(payload['args'])
            )
            kwargs = cloudpickle.loads(
                base64.b64decode(payload['kwargs'])
            )
            return func, args, kwargs
        except Exception as e:
            raise ValueError(f"Failed to deserialize payload: {str(e)}")

    @staticmethod
    def serialize_result(result: Any) -> str:
        return base64.b64encode(
            cloudpickle.dumps(result)
        ).decode('utf-8')

class Handler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing handler with device: {self.device}")

    def handle_job(self, job) -> Dict:
        """Handle a Runpod job"""
        job_input = job['input']
        task_id = job_input.get('task_id', 'unknown')
        
        with task_execution_context(task_id) as stats:
            try:
                # Deserialize and execute
                func, args, kwargs = PayloadProcessor.deserialize_payload(job_input)
                
                # Move tensors to correct device if needed
                args = self._prepare_tensors(args)
                kwargs = self._prepare_tensors(kwargs)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Prepare result
                result = self._prepare_result(result)
                
                return {
                    'output': PayloadProcessor.serialize_result(result),
                    'stats': {
                        'duration': stats.duration,
                        'gpu_memory': stats.gpu_memory_used,
                        'cpu_memory': stats.cpu_memory_used
                    }
                }
                
            except Exception as e:
                stats.error = str(e)
                logger.error(f"Task {task_id} failed: {str(e)}")
                logger.error(traceback.format_exc())
                
                return {
                    'error': {
                        'message': str(e),
                        'type': type(e).__name__,
                        'traceback': traceback.format_exc()
                    },
                    'stats': {
                        'duration': stats.duration,
                        'gpu_memory': stats.gpu_memory_used,
                        'cpu_memory': stats.cpu_memory_used
                    }
                }

    def _prepare_tensors(self, data: Any) -> Any:
        """Recursively move tensors to correct device"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._prepare_tensors(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._prepare_tensors(x) for x in data)
        return data

    def _prepare_result(self, result: Any) -> Any:
        """Prepare result for serialization"""
        if isinstance(result, torch.Tensor):
            return result.cpu()
        elif isinstance(data, dict):
            return {k: self._prepare_result(v) for k, v in result.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._prepare_result(x) for x in result)
        return result

# Initialize handler and start serverless
handler = Handler()
runpod.serverless.start({
    "handler": handler.handle_job
})
