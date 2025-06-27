from dotenv import load_dotenv
import os

# Load environment variables from .env file if it exists
load_dotenv()

# Import app after loading environment variables
from faq_bot import app

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8980))
    app.run(host='0.0.0.0', port=port)
