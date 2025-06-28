import os
import sys

# Add the project directory to the path
path = os.path.dirname(os.path.abspath(__file__))
if path not in sys.path:
    sys.path.append(path)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(path, '.env'))

# Import the Flask app
from faq_bot import app as application  # noqa
