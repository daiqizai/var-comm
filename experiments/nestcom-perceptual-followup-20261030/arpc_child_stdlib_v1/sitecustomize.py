"""Prebind only stdlib statistics in ARPC compiler child interpreters."""
import importlib.util
import sys
import sysconfig
from pathlib import Path

path = Path(sysconfig.get_path('stdlib')) / 'statistics.py'
spec = importlib.util.spec_from_file_location('statistics', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules['statistics'] = module
