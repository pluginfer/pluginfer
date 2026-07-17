"""
Verification Suite for World-Class Features
Tests System Doctor, Auto-Updater, Payments, and GUI headless load.
"""
import unittest
import sys
import os
import time

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.system_doctor import SystemDoctor
from core.updater import AutoUpdater
from core.payments import StripeMockGateway
from core.mesh_controller import MeshNetworkController

class WorldClassRefactorTest(unittest.TestCase):
    
    def test_auto_updater(self):
        print("\n[Test] Auto-Updater...")
        updater = AutoUpdater(current_version="0.9.0")
        update_info = updater.check_for_updates()
        
        self.assertIsNotNone(update_info, "Updater should find mock update")
        self.assertEqual(update_info['version'], "1.0.0")
        print("✅ Updater detected new version 1.0.0")
        
    def test_system_doctor(self):
        print("\n[Test] System Doctor...")
        doctor = SystemDoctor()
        
        # Test Diagnostics
        report = doctor.run_diagnostics()
        print(f"   Diagnostic Report: {report}")
        
        self.assertIn('network', report)
        self.assertIn('disk', report)
        self.assertIn('issues', report)
        print("✅ Diagnostics ran successfully")
        
        # Test Repair Logic (Simulated)
        doctor.health_status['issues'] = ['no_internet'] # Force an issue
        doctor._attempt_repairs()
        
        self.assertTrue(len(doctor.repair_history) > 0)
        self.assertEqual(doctor.repair_history[0]['issue'], 'no_internet')
        print("✅ Doctor attempted network repair")
        
    def test_payments(self):
        print("\n[Test] Payment Gateway...")
        gateway = StripeMockGateway()
        
        user = "test_user_123"
        success = gateway.process_payment(user, 10.0, "Test Charge")
        self.assertTrue(success)
        
        balance = gateway.get_balance(user)
        self.assertEqual(balance, -10.0) # Check debit
        print("✅ Payment processed and ledger updated")
        
    def test_gui_load(self):
        print("\n[Test] GUI Import...")
        try:
            from ui.desktop_app import PluginferApp
            import tkinter
            print("✅ GUI Module imports successfully")
        except ImportError as e:
            self.fail(f"GUI Import failed: {e}")

if __name__ == '__main__':
    unittest.main()
