"""
License Validator
Manages license validation, feature gating, and usage tracking
"""
import hashlib
import json
import logging
import platform
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)

class LicenseTier:
    """License tier definitions"""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class LicenseValidator:
    """
    Validates licenses and manages feature access based on tier.
    
    Features by tier:
    - FREE: CPU only, 100 inferences/day, single plugin
    - PRO: GPU support, unlimited inferences, multiple plugins, QAL
    - ENTERPRISE: All features, multi-GPU, clustering, priority support
    """
    
    TIER_FEATURES = {
        LicenseTier.FREE: {
            'gpu_support': False,
            'max_daily_inferences': 100,
            'max_plugins': 1,
            'qal_enabled': False,
            'multi_gpu': False,
            'clustering': False,
            'batch_size': 1
        },
        LicenseTier.PRO: {
            'gpu_support': True,
            'max_daily_inferences': -1,  # Unlimited
            'max_plugins': 10,
            'qal_enabled': True,
            'multi_gpu': False,
            'clustering': False,
            'batch_size': 32
        },
        LicenseTier.ENTERPRISE: {
            'gpu_support': True,
            'max_daily_inferences': -1,
            'max_plugins': -1,  # Unlimited
            'qal_enabled': True,
            'multi_gpu': True,
            'clustering': True,
            'batch_size': 128
        }
    }
    
    def __init__(self, license_file: str = "license.json"):
        self.license_file = Path(license_file)
        self.license_data: Optional[Dict] = None
        self.device_fingerprint = self._generate_device_fingerprint()
        self.usage_count = 0
        self.usage_file = Path("usage.json")
        
        self._load_license()
        self._load_usage()
    
    def _generate_device_fingerprint(self) -> str:
        """
        Generate a unique device fingerprint.
        Combines hardware info to create a stable identifier.
        """
        try:
            # Get system info
            system_info = {
                'platform': platform.system(),
                'machine': platform.machine(),
                'processor': platform.processor(),
            }
            
            # Try to get MAC address
            try:
                mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff)
                              for elements in range(0,2*6,2)][::-1])
                system_info['mac'] = mac
            except:
                pass
            
            # Try to get disk serial (Linux/Mac)
            try:
                if platform.system() == 'Linux':
                    result = subprocess.run(['lsblk', '-o', 'SERIAL'], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        serial = result.stdout.strip().split('\n')[1]
                        system_info['disk_serial'] = serial
            except:
                pass
            
            # Create hash
            info_str = json.dumps(system_info, sort_keys=True)
            fingerprint = hashlib.sha256(info_str.encode()).hexdigest()[:16]
            
            return fingerprint
            
        except Exception as e:
            logger.warning(f"Could not generate device fingerprint: {e}")
            return "unknown"
    
    def _load_license(self):
        """Load license from file"""
        if not self.license_file.exists():
            logger.info("No license file found - using FREE tier")
            self.license_data = {
                'tier': LicenseTier.FREE,
                'key': 'FREE-LICENSE',
                'valid_until': None,
                'device_fingerprint': None
            }
            return
        
        try:
            with open(self.license_file, 'r') as f:
                self.license_data = json.load(f)
            logger.info(f"Loaded license: {self.license_data['tier'].upper()}")
        except Exception as e:
            logger.error(f"Failed to load license: {e}")
            self.license_data = {
                'tier': LicenseTier.FREE,
                'key': 'FREE-LICENSE',
                'valid_until': None
            }
    
    def _load_usage(self):
        """Load usage tracking data"""
        if not self.usage_file.exists():
            self._reset_usage()
            return
        
        try:
            with open(self.usage_file, 'r') as f:
                usage_data = json.load(f)
            
            # Check if it's a new day
            last_date = datetime.fromisoformat(usage_data.get('date', '2000-01-01'))
            today = datetime.now().date()
            
            if last_date.date() != today:
                self._reset_usage()
            else:
                self.usage_count = usage_data.get('count', 0)
                
        except Exception as e:
            logger.error(f"Failed to load usage data: {e}")
            self._reset_usage()
    
    def _reset_usage(self):
        """Reset daily usage counter"""
        self.usage_count = 0
        self._save_usage()
    
    def _save_usage(self):
        """Save usage data"""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump({
                    'date': datetime.now().isoformat(),
                    'count': self.usage_count
                }, f)
        except Exception as e:
            logger.error(f"Failed to save usage data: {e}")
    
    def validate(self) -> bool:
        """
        Validate the current license.
        
        Returns:
            True if license is valid
        """
        if not self.license_data:
            return False
        
        # Check expiry
        if self.license_data.get('valid_until'):
            expiry = datetime.fromisoformat(self.license_data['valid_until'])
            if datetime.now() > expiry:
                logger.error("License has expired")
                return False
        
        # Check device binding (for PRO and ENTERPRISE)
        tier = self.license_data.get('tier', LicenseTier.FREE)
        if tier != LicenseTier.FREE:
            bound_fingerprint = self.license_data.get('device_fingerprint')
            if bound_fingerprint and bound_fingerprint != self.device_fingerprint:
                logger.error("License is bound to a different device")
                return False
        
        return True
    
    def check_feature(self, feature: str) -> bool:
        """
        Check if a feature is available in current tier.
        
        Args:
            feature: Feature name to check
            
        Returns:
            True if feature is available
        """
        if not self.validate():
            return False
        
        tier = self.license_data.get('tier', LicenseTier.FREE)
        features = self.TIER_FEATURES.get(tier, self.TIER_FEATURES[LicenseTier.FREE])
        
        return features.get(feature, False)
    
    def get_feature_value(self, feature: str) -> Any:
        """Get the value of a feature limit"""
        tier = self.license_data.get('tier', LicenseTier.FREE)
        features = self.TIER_FEATURES.get(tier, self.TIER_FEATURES[LicenseTier.FREE])
        return features.get(feature)
    
    def record_inference(self) -> bool:
        """
        Record an inference execution and check quota.
        
        Returns:
            True if inference is allowed
        """
        if not self.validate():
            return False
        
        max_daily = self.get_feature_value('max_daily_inferences')
        
        if max_daily == -1:  # Unlimited
            self.usage_count += 1
            self._save_usage()
            return True
        
        if self.usage_count >= max_daily:
            logger.error(f"Daily inference quota exceeded ({max_daily})")
            return False
        
        self.usage_count += 1
        self._save_usage()
        return True
    
    def get_tier(self) -> str:
        """Get current license tier"""
        return self.license_data.get('tier', LicenseTier.FREE)
    
    def get_usage_info(self) -> Dict[str, Any]:
        """Get usage information"""
        max_daily = self.get_feature_value('max_daily_inferences')
        
        return {
            'tier': self.get_tier(),
            'daily_usage': self.usage_count,
            'daily_limit': max_daily if max_daily != -1 else 'unlimited',
            'device_fingerprint': self.device_fingerprint
        }
    
    def print_license_info(self):
        """Print license information"""
        if not self.license_data:
            print("⚠️  No valid license")
            return
        
        tier = self.license_data.get('tier', LicenseTier.FREE)
        features = self.TIER_FEATURES.get(tier, self.TIER_FEATURES[LicenseTier.FREE])
        
        print("\n" + "="*60)
        print("🔐 LICENSE INFORMATION")
        print("="*60)
        print(f"\nTier: {tier.upper()}")
        print(f"Key: {self.license_data.get('key', 'N/A')}")
        
        if self.license_data.get('valid_until'):
            expiry = datetime.fromisoformat(self.license_data['valid_until'])
            print(f"Valid Until: {expiry.strftime('%Y-%m-%d')}")
        else:
            print("Valid Until: Perpetual")
        
        print(f"\n📊 Usage:")
        usage = self.get_usage_info()
        print(f"  Today: {usage['daily_usage']} / {usage['daily_limit']}")
        
        print(f"\n✨ Features:")
        print(f"  GPU Support: {'✅' if features['gpu_support'] else '❌'}")
        print(f"  QAL Enabled: {'✅' if features['qal_enabled'] else '❌'}")
        print(f"  Multi-GPU: {'✅' if features['multi_gpu'] else '❌'}")
        print(f"  Max Plugins: {features['max_plugins'] if features['max_plugins'] != -1 else 'Unlimited'}")
        print(f"  Batch Size: {features['batch_size']}")
        
        print(f"\n🔑 Device: {self.device_fingerprint}")
        print("="*60 + "\n")


def generate_license(tier: str, key: str, valid_days: int = 365, 
                    device_fingerprint: Optional[str] = None) -> Dict:
    """
    Generate a license file.
    
    Args:
        tier: License tier (free, pro, enterprise)
        key: License key string
        valid_days: Number of days license is valid
        device_fingerprint: Optional device binding
        
    Returns:
        License dictionary
    """
    license_data = {
        'tier': tier,
        'key': key,
        'valid_until': (datetime.now() + timedelta(days=valid_days)).isoformat(),
        'device_fingerprint': device_fingerprint,
        'generated_at': datetime.now().isoformat()
    }
    
    return license_data


if __name__ == "__main__":
    validator = LicenseValidator()
    validator.print_license_info()
