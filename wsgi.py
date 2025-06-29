import os
import sys
from pathlib import Path

# Add the project directory to the path
path = Path(__file__).parent.absolute()
sys.path.insert(0, str(path))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(path / '.env')

# Import the Flask app
from faq_bot import app

# Create application object for Gunicorn
application = app
