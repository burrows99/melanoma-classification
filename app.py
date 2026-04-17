import sys
from pathlib import Path

# Add src/ to Python path so imports work as in the main repo
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app import App

App().launch()
