import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Create necessary directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/competitions', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)
os.makedirs('examples', exist_ok=True)
os.makedirs('tests', exist_ok=True)

print("Forecasting project initialized!")
print("Created necessary directories.")
print("\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Run examples: python examples/basic_forecasting.py")
print("3. Check documentation: docs/README.md")

