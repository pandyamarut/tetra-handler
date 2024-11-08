# advanced_handler.py
import runpod
import traceback
import logging
import time
import torch
import psutil
import gc
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager


import base64
import cloudpickle

def handler(job):
    """Runpod serverless handler"""
    try:
        # Extract input data
        input_data: Dict = job['input']
        
        # Deserialize function and arguments
        func = cloudpickle.loads(
            base64.b64decode(input_data['function'].encode('utf-8'))
        )
        args = cloudpickle.loads(
            base64.b64decode(input_data['args'].encode('utf-8'))
        )
        kwargs = cloudpickle.loads(
            base64.b64decode(input_data['kwargs'].encode('utf-8'))
        )
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Serialize result
        output = base64.b64encode(cloudpickle.dumps(result)).decode('utf-8')
        
        return {
            "output": output
        }
        
    except Exception as e:
        return {
            "error": str(e)
        }
runpod.serverless.start({
    "handler": handler
})
