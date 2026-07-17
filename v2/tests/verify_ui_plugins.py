
import sys
import os
import unittest
import json
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.professional_web_ui import app
import ui.professional_web_ui as ui_module

class TestUIPlugins(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.mock_controller = MagicMock()
        ui_module.mesh_controller = self.mock_controller
        
        # Mock Plugin Registry
        self.mock_controller.plugin_registry.list_plugins.return_value = ['img_resize', 'txt_sentiment', 'super_new_plugin']
        
    def test_plugin_api_lists_correctly(self):
        """Ensure /api/plugins returns the dynamic list from registry"""
        print("\n[TEST] Calling /api/plugins...")
        resp = self.app.get('/api/plugins')
        data = json.loads(resp.data)
        
        print(f"Plugins Found: {data.get('plugins')}")
        self.assertIn('super_new_plugin', data['plugins'])
        self.assertEqual(len(data['plugins']), 3)
        print("✅ Plugin API lists dynamically loaded plugins.")

    def test_client_portal_loads(self):
        """Ensure /client route loads 200 OK (Template rendering)"""
        print("\n[TEST] Loading Client Portal...")
        resp = self.app.get('/client')
        self.assertEqual(resp.status_code, 200)
        print("✅ Client Portal Loaded Successfully.")

    def test_submit_job_wiring(self):
        """Ensure submit_task calls the controller"""
        print("\n[TEST] Submitting Job...")
        self.mock_controller.submit_task.return_value = 'task_123'
        
        resp = self.app.post('/api/submit_job', data={
            'plugin': 'super_new_plugin',
            'text': 'hello'
        })
        
        data = json.loads(resp.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['task_id'], 'task_123')
        
        # Verify wiring
        self.mock_controller.submit_task.assert_called_once()
        print("✅ Job Submission Wired correctly.")

if __name__ == '__main__':
    unittest.main()
