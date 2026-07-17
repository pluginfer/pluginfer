from core.plugin_base import PluginBase
import torch
import time
import logging

logger = logging.getLogger(__name__)

class SmartTensorOps(PluginBase):
    def config(self):
        return {
            'name': 'Smart Tensor Ops',
            'version': '1.0.0',
            'description': 'Demonstrates hardware-optimized tensor operations (GPU/DirectML)',
            'category': 'benchmark'
        }

    def run(self, input_data):
        """
        Implementation of abstract run method.
        Delegates to execute with default device.
        """
        return self.execute(input_data, device=None)
        
    def execute(self, input_data, device=None):
        """
        Executes matrix multiplication on the optimized device.
        Input: {'size': 2048, 'iterations': 10}
        """
        size = input_data.get('size', 2048) # Default 2048
        iterations = input_data.get('iterations', 10)
        
        # Fallback if device not passed
        if device is None:
            device = torch.device('cpu')
            
        logger.info(f"Allocating {size}x{size} tensors on {device}...")
        
        try:
            # Create tensors directly on device
            # This is the "Optimization" - using the correct backend memory
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            logger.info("Starting optimized computation loop...")
            start = time.time()
            
            for i in range(iterations):
                c = torch.matmul(a, b)
                
                # Synchronization for accurate timing (GPU is async)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                # For DirectML, operations are generally queued but we don't have a generic sync command
                # readily exposed in simple torch API without dml specific calls, 
                # but standard execution flow usually keeps it reasonable for benchmarking.
                
            elapsed = time.time() - start
            
            # Theoretical FLOPS for MatMul: 2 * n^3
            ops = 2 * (size ** 3) * iterations
            tflops = ops / (elapsed * 1e12)
            
            logger.info(f"Completed in {elapsed:.4f}s ({tflops:.2f} TFLOPS)")
            
            return {
                'status': 'success',
                'device': str(device),
                'device_type': device.type,
                'matrix_size': size,
                'iterations': iterations,
                'total_time': elapsed,
                'avg_time_per_run': elapsed / iterations,
                'tflops': tflops,
                'message': f"Optimized execution on {device} completed successfully."
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'device': str(device)
            }
