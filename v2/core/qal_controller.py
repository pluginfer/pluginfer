"""
Quantum Acceleration Layer (QAL)
High-level workload distribution and device management
"""
import logging
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .plugin_base import PluginBase
from .hardware_detector import HardwareDetector

logger = logging.getLogger(__name__)

class QALController:
    """
    Quantum Acceleration Layer - manages distributed workload execution.
    
    Phase 1 Implementation:
    - Device detection and profiling
    - High-level batch splitting
    - Parallel execution coordination
    - Performance monitoring
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.hardware = HardwareDetector()
        self.devices = self.hardware.detect_all_devices()
        self.max_workers = max_workers or len(self.devices)
        self.execution_log: List[Dict] = []
        
        logger.info(f"QAL initialized with {len(self.devices)} devices")
    
    def distribute_workload(self, plugin: PluginBase, 
                          input_data_list: List[Dict[str, Any]],
                          strategy: str = 'auto') -> List[Dict[str, Any]]:
        """
        Distribute workload across available devices.
        
        Args:
            plugin: Plugin to execute
            input_data_list: List of inputs to process
            strategy: Distribution strategy ('auto', 'round_robin', 'fastest')
            
        Returns:
            List of results
        """
        start_time = time.time()
        
        if len(input_data_list) == 1:
            # Single input - just run on best device
            return [self._execute_on_device(plugin, input_data_list[0], self.devices[0])]
        
        logger.info(f"Distributing {len(input_data_list)} tasks using '{strategy}' strategy")
        
        if strategy == 'auto':
            results = self._auto_distribute(plugin, input_data_list)
        elif strategy == 'round_robin':
            results = self._round_robin_distribute(plugin, input_data_list)
        elif strategy == 'fastest':
            results = self._fastest_first_distribute(plugin, input_data_list)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        total_time = time.time() - start_time
        
        logger.info(f"QAL completed {len(results)} tasks in {total_time:.2f}s")
        
        return results
    
    def _auto_distribute(self, plugin: PluginBase, 
                        input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Automatically distribute based on device capabilities.
        Uses available GPUs first, then falls back to CPU parallelization.
        """
        gpu_devices = [d for d in self.devices if d['type'] in ['cuda', 'mps', 'rocm']]
        
        if gpu_devices and 'count' in gpu_devices[0] and gpu_devices[0]['count'] > 1:
            # Multiple GPUs available - distribute across them
            return self._multi_gpu_distribute(plugin, input_data_list, gpu_devices[0]['count'])
        
        elif gpu_devices:
            # Single GPU - batch process on GPU
            return self._batch_on_device(plugin, input_data_list, gpu_devices[0])
        
        else:
            # CPU only - use parallel processing
            return self._parallel_cpu_distribute(plugin, input_data_list)
    
    def _multi_gpu_distribute(self, plugin: PluginBase, 
                             input_data_list: List[Dict[str, Any]], 
                             gpu_count: int) -> List[Dict[str, Any]]:
        """Distribute across multiple GPUs"""
        logger.info(f"Distributing across {gpu_count} GPUs")
        
        # Split inputs across GPUs
        batch_size = len(input_data_list) // gpu_count
        results = [None] * len(input_data_list)
        
        with ThreadPoolExecutor(max_workers=gpu_count) as executor:
            futures = []
            
            for gpu_id in range(gpu_count):
                start_idx = gpu_id * batch_size
                end_idx = start_idx + batch_size if gpu_id < gpu_count - 1 else len(input_data_list)
                batch = input_data_list[start_idx:end_idx]
                
                device_info = {
                    'type': 'cuda',
                    'name': f'GPU {gpu_id}',
                    'id': gpu_id
                }
                
                future = executor.submit(
                    self._process_batch, 
                    plugin, batch, device_info, start_idx
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                batch_results, start_idx = future.result()
                for i, result in enumerate(batch_results):
                    results[start_idx + i] = result
        
        return results
    
    def _batch_on_device(self, plugin: PluginBase, 
                        input_data_list: List[Dict[str, Any]], 
                        device: Dict) -> List[Dict[str, Any]]:
        """Process all inputs sequentially on a single device"""
        logger.info(f"Batch processing on {device['name']}")
        
        results = []
        for input_data in input_data_list:
            result = self._execute_on_device(plugin, input_data, device)
            results.append(result)
        
        return results
    
    def _parallel_cpu_distribute(self, plugin: PluginBase, 
                                input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Distribute across CPU cores using threading"""
        logger.info(f"Parallel CPU processing with {self.max_workers} workers")
        
        cpu_device = [d for d in self.devices if d['type'] == 'cpu'][0]
        results = [None] * len(input_data_list)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._execute_on_device, plugin, input_data, cpu_device): idx
                for idx, input_data in enumerate(input_data_list)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        
        return results
    
    def _round_robin_distribute(self, plugin: PluginBase, 
                               input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Distribute using round-robin across devices"""
        logger.info("Round-robin distribution across devices")
        
        results = []
        for i, input_data in enumerate(input_data_list):
            device = self.devices[i % len(self.devices)]
            result = self._execute_on_device(plugin, input_data, device)
            results.append(result)
        
        return results
    
    def _fastest_first_distribute(self, plugin: PluginBase, 
                                 input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Always use the fastest device"""
        logger.info("Using fastest device for all tasks")
        
        fastest_device = self.devices[0]  # Already sorted by priority
        return self._batch_on_device(plugin, input_data_list, fastest_device)
    
    def _execute_on_device(self, plugin: PluginBase, input_data: Dict[str, Any], 
                          device: Dict) -> Dict[str, Any]:
        """Execute a single task on a specific device"""
        start_time = time.time()
        
        try:
            result = plugin.execute(input_data, device=device['type'])
            result['_metadata']['qal_device'] = device['name']
            
            execution_time = time.time() - start_time
            self._log_execution(plugin.name, device, execution_time, 'success')
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._log_execution(plugin.name, device, execution_time, 'failed')
            
            return {
                'error': str(e),
                '_metadata': {
                    'plugin': plugin.name,
                    'qal_device': device['name'],
                    'execution_time': execution_time,
                    'status': 'failed'
                }
            }
    
    def _process_batch(self, plugin: PluginBase, batch: List[Dict], 
                      device: Dict, start_idx: int) -> tuple:
        """Process a batch of inputs on a specific device"""
        results = []
        for input_data in batch:
            result = self._execute_on_device(plugin, input_data, device)
            results.append(result)
        return results, start_idx
    
    def _log_execution(self, plugin_name: str, device: Dict, 
                      execution_time: float, status: str):
        """Log execution details"""
        self.execution_log.append({
            'plugin': plugin_name,
            'device': device['name'],
            'device_type': device['type'],
            'time': execution_time,
            'status': status,
            'timestamp': time.time()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all devices"""
        if not self.execution_log:
            return {'executions': 0}
        
        total = len(self.execution_log)
        by_device = {}
        
        for log in self.execution_log:
            device = log['device']
            if device not in by_device:
                by_device[device] = {
                    'count': 0,
                    'total_time': 0,
                    'successful': 0,
                    'failed': 0
                }
            
            by_device[device]['count'] += 1
            by_device[device]['total_time'] += log['time']
            
            if log['status'] == 'success':
                by_device[device]['successful'] += 1
            else:
                by_device[device]['failed'] += 1
        
        # Calculate averages
        for device, stats in by_device.items():
            stats['average_time'] = stats['total_time'] / stats['count']
        
        return {
            'total_executions': total,
            'devices': by_device
        }
    
    def print_performance_summary(self):
        """Print performance summary"""
        summary = self.get_performance_summary()
        
        print("\n" + "="*60)
        print("⚡ QAL PERFORMANCE SUMMARY")
        print("="*60)
        
        if summary['total_executions'] == 0:
            print("No executions recorded.")
        else:
            print(f"\nTotal Executions: {summary['total_executions']}")
            print("\n🖥️  By Device:")
            
            for device, stats in summary['devices'].items():
                print(f"\n  {device}:")
                print(f"    Executions: {stats['count']}")
                print(f"    Successful: {stats['successful']}")
                print(f"    Failed: {stats['failed']}")
                print(f"    Avg Time: {stats['average_time']:.4f}s")
                print(f"    Total Time: {stats['total_time']:.2f}s")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    qal = QALController()
    qal.print_performance_summary()
