"""
Plugin Base Class
All plugins must inherit from this base class
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time

class PluginBase(ABC):
    """
    Abstract base class for all Pluginfer plugins.
    
    Each plugin must implement:
    - run(): Execute the main inference/processing
    - config(): Return plugin metadata
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.execution_count = 0
        self.total_time = 0.0
        
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the plugin's main functionality.
        
        Args:
            input_data: Dictionary containing input data
            
        Returns:
            Dictionary containing results and metadata
        """
        pass
    
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """
        Return plugin configuration and metadata.
        
        Returns:
            Dictionary with plugin info (name, version, description, etc.)
        """
        pass
    
    def execute(self, input_data: Dict[str, Any], device: Optional[Any] = None) -> Dict[str, Any]:
        """
        Wrapper that measures execution time and adds metadata.
        
        Args:
            input_data: Input data for the plugin
            device: Optional compute device to use
            
        Returns:
            Result dictionary with added metadata
        """
        start_time = time.time()
        
        try:
            # Run the actual plugin logic
            result = self.run(input_data)
            
            # Add execution metadata
            execution_time = time.time() - start_time
            self.execution_count += 1
            self.total_time += execution_time
            
            result['_metadata'] = {
                'plugin': self.name,
                'execution_time': execution_time,
                'execution_count': self.execution_count,
                'average_time': self.total_time / self.execution_count,
                'device': str(device) if device else 'default',
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                'error': str(e),
                '_metadata': {
                    'plugin': self.name,
                    'execution_time': execution_time,
                    'status': 'failed',
                    'error_type': type(e).__name__
                }
            }
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: list) -> bool:
        """
        Helper to validate required input fields.
        
        Args:
            input_data: Input dictionary
            required_fields: List of required field names
            
        Returns:
            True if all required fields present
            
        Raises:
            ValueError: If required fields are missing
        """
        missing = [field for field in required_fields if field not in input_data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics for this plugin"""
        return {
            'name': self.name,
            'execution_count': self.execution_count,
            'total_time': self.total_time,
            'average_time': self.total_time / self.execution_count if self.execution_count > 0 else 0
        }
    
    def __repr__(self):
        return f"<Plugin: {self.name}>"
