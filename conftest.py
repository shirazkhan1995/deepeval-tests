from pathlib import Path
from dotenv import load_dotenv

# Load environment before any deepeval imports touch OpenAI
load_dotenv(Path(__file__).parent / ".env.development", override=True)
