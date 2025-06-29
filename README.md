# Jupiter Money FAQ Bot

This project implements a FAQ bot for Jupiter Money that can answer user queries by retrieving relevant information from Jupiter Money's website. The system uses web scraping, natural language processing, and vector similarity search to provide accurate responses.

## Features

- Extracts URLs from Jupiter Money's website
- Crawls and processes web content
- Generates summaries of web pages using Ollama
- Stores embeddings in Pinecone for efficient similarity search
- Provides a query interface to get answers to user questions

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Ollama (local LLM)
- Pinecone account (for vector database)

## Web Interface

The FAQ bot comes with a modern web interface that can be accessed at `http://localhost:8980` when running locally.

### Local Development

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your environment variables in a `.env` file:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   ```

3. Run the development server:
   ```bash
   python faq_bot.py
   ```

4. Open your browser and navigate to `http://localhost:8980`

## Deployment

### Deploying to Render.com

1. **Create a Render.com account**
   - Go to [render.com](https://render.com/) and sign up if you don't have an account

2. **Create a new Web Service**
   - Click "New" and select "Web Service"
   - Connect your GitHub/GitLab repository or use the Render CLI

3. **Configure your service**
   - Name: `jupiter-faq-bot`
   - Region: Choose the one closest to your users
   - Branch: `main` (or your preferred branch)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn faq_bot:app`

4. **Set up environment variables**
   - `PYTHON_VERSION`: `3.10.13`
   - `PINECONE_API_KEY`: Your Pinecone API key
   - `PINECONE_INDEX_NAME`: Your Pinecone index name
   - `FLASK_DEBUG`: `false` (for production)

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

6. **Access your application**
   - Once deployed, you'll get a URL like `https://jupiter-faq-bot.onrender.com`

### Environment Variables

For production, make sure to set these environment variables:

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Your Pinecone index name
- `PORT`: (Optional) Port to run the application on (default: 8980)
- `FLASK_DEBUG`: Set to `false` in production

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/git-Adi/Jupiter_bot.git

   ```

2. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**

   ```bash
   pip install -r requirements_faq_bot.txt
   pip install -U crawl4ai --pre
   pip install pinecone
   pip install ollama
   ```

4. **Set up Ollama**

   ```bash
   # Install Ollama (follow instructions at https://ollama.ai/)
   # Run post-installation setup
   crawl4ai-setup

   # Verify installation
   crawl4ai-doctor
   ```

5. **Set up environment variables**
   Create a `.env` file in the project root with your Pinecone credentials:

   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   ```

## Usage

### 1. Extract URLs

Run the Jupyter notebook to extract URLs from Jupiter Money's website:

```bash
jupyter notebook link.ipynb
```

This will save the URLs to `extracted_urls.txt`.

### 2. Process URLs and Generate Summaries

#### For all URLs:

```bash
python crawl4ai/ollama_only_summarizer.py
```

This processes all URLs from `extracted_urls.txt` and saves the content to `final_content.json`.

#### For specific URLs:

```bash
python crawl4ai/ollama_only_summarizer_new.py
```

This processes specific policy-related URLs and saves the content to `final_content1.json`.

### 3. Upload to Pinecone

```bash
python upsert_to_pinecone.py
```

This uploads the processed content to Pinecone for similarity search.

### 4. Run the FAQ Bot

```bash
python faq_bot.py
```

This starts the FAQ bot interface where you can ask questions about Jupiter Money's services.

## Project Structure

- `link.ipynb`: Jupyter notebook to extract URLs from Jupiter Money's website
- `extracted_urls.txt`: List of extracted URLs
- `crawl4ai/`: Contains scripts for web crawling and content processing
  - `ollama_only_summarizer.py`: Processes all URLs and generates summaries
  - `ollama_only_summarizer_new.py`: Processes specific policy-related URLs
- `final_content.json`: Processed content from all URLs
- `final_content1.json`: Processed content from policy-related URLs
- `upsert_to_pinecone.py`: Script to upload content to Pinecone
- `faq_bot.py`: Main script to query the FAQ system
- `requirements_faq_bot.txt`: Python dependencies

## Dependencies

- `sentence-transformers>=2.2.2`: For generating embeddings
- `numpy>=1.24.0`: For numerical operations
- `langdetect>=1.0.9`: For language detection
- `translate>=3.6.1`: For text translation
- `python-dotenv>=1.0.0`: For managing environment variables
- `crawl4ai`: For web crawling and content extraction
- `pinecone-client`: For vector database operations
- `ollama`: For running local language models

## Troubleshooting

1. **Ollama not found**

   - Ensure Ollama is installed and running
   - Verify it's in your system PATH

2. **Pinecone connection issues**

   - Check your API key and environment in the `.env` file
   - Verify your Pinecone index exists and is accessible

3. **Missing dependencies**

   - Make sure all required packages are installed
   - Try running `pip install -r requirements_faq_bot.txt`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Jupiter Money for the public information
- Ollama for the local language model
- Pinecone for vector database services
