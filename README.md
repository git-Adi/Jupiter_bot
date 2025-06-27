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

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd jupiter_intern
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
python crawl4ai/crawl4ai/ollama_only_summarizer.py
```

This processes all URLs from `extracted_urls.txt` and saves the content to `final_content.json`.

#### For specific URLs:

```bash
python crawl4ai/crawl4ai/ollama_only_summarizer_new.py
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
