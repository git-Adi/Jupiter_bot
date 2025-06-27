import asyncio
import textwrap
from typing import Dict, Any
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from autogen import AssistantAgent, UserProxyAgent

class OllamaSummarizer:
    def __init__(self, model_name: str = "llama3"):
        """
        Initialize the Ollama summarizer with a specific model.
        
        Args:
            model_name (str): The name of the Ollama model to use (default: "llama3")
        """
        self.model_name = model_name
        self.config_list = [
            {
                "model": model_name,  # Just use the model name without 'ollama/'
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",  # ollama doesn't use an API key
                "api_type": "ollama"  # Changed from 'open_ai' to 'ollama'
            }
        ]
        
        # Initialize AutoGen agents
        self.assistant = AssistantAgent(
            name="assistant",
            llm_config={
                "config_list": self.config_list,
                "temperature": 0.3,  # Lower temperature for more focused summaries
            }
        )
        
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            code_execution_config=False,
        )
    
    async def extract_content(self, url: str) -> str:
        """
        Extract text content from a URL using crawl4ai.
        
        Args:
            url (str): The URL to extract content from
            
        Returns:
            str: Extracted text content
        """
        async with AsyncWebCrawler() as crawler:
            response = await crawler.arun(url=url)
            soup = BeautifulSoup(response.html, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator='\n', strip=True)
    
    async def summarize(self, content: str) -> Dict[str, Any]:
        """
        Generate a summary of the provided content using Ollama via AutoGen.
        
        Args:
            content (str): The content to summarize
            
        Returns:
            Dict[str, Any]: Dictionary containing the summary and metadata
        """
        # Limit content length to avoid token limits
        content = content[:12000]  # Adjust based on your model's context window
        
        summary_prompt = f"""
        Please analyze the following content and provide a structured summary including:
        1. A concise title (less than 10 words)
        2. A brief summary (2-3 sentences)
        3. 3-5 key points as bullet points
        4. Overall sentiment (positive/neutral/negative)
        
        Format your response as a JSON object with the following structure:
        {{
            "title": "...",
            "summary": "...",
            "key_points": ["...", "...", "..."],
            "sentiment": "..."
        }}
        
        Content to summarize:
        {content}
        """
        
        # Create a new event loop for the synchronous AutoGen call
        loop = asyncio.get_event_loop()
        
        # Run the synchronous AutoGen code in a thread
        def run_autogen_chat():
            self.user_proxy.initiate_chat(
                self.assistant,
                message=summary_prompt,
                summary_method="last_msg"
            )
            return self.user_proxy.chat_messages[self.assistant][-1]['content']
            
        # Execute the synchronous code in a thread
        last_message = await loop.run_in_executor(None, run_autogen_chat)
        
        # Try to parse the JSON response from the model's output
        try:
            import json
            import re
            
            # First, try to find a JSON block in the response
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', last_message, re.DOTALL)
            
            if not json_match:
                # If no code block, look for a JSON object directly
                json_match = re.search(r'({[\s\S]*})', last_message)
            
            if json_match:
                json_str = json_match.group(1).strip()
                # Clean up the JSON string
                json_str = re.sub(r'^```(?:json)?|```$', '', json_str, flags=re.MULTILINE).strip()
                
                # Parse the JSON
                summary_data = json.loads(json_str)
                
                # Ensure all required fields are present
                if not all(key in summary_data for key in ["title", "summary", "key_points", "sentiment"]):
                    raise ValueError("Missing required fields in JSON response")
                    
                return summary_data
            else:
                # If no JSON found, try to extract structured data from the text
                title_match = re.search(r'"?title"?\s*[:=]\s*"?([^"]+)"?', last_message, re.IGNORECASE)
                summary_match = re.search(r'"?summary"?\s*[:=]\s*"?([^"]+)"?', last_message, re.IGNORECASE | re.DOTALL)
                
                key_points = re.findall(r'[•*-]\s*([^\n]+)', last_message) or \
                             re.findall(r'\d+\.\s*([^\n]+)', last_message) or \
                             ["No key points extracted"]
                
                sentiment = "neutral"
                if re.search(r'sentiment.*positive', last_message, re.IGNORECASE):
                    sentiment = "positive"
                elif re.search(r'sentiment.*negative', last_message, re.IGNORECASE):
                    sentiment = "negative"
                
                return {
                    "title": title_match.group(1).strip('"') if title_match else "Summary",
                    "summary": summary_match.group(1).strip('"') if summary_match else last_message.split('\n')[0][:200],
                    "key_points": key_points[:5],  # Limit to 5 key points
                    "sentiment": sentiment,
                    "raw_response": last_message,
                    "note": "Extracted from unstructured response"
                }
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing JSON response: {e}")
            print("Raw response:", last_message)
            
            # Fallback to returning a structured response with the raw content
            return {
                "title": "Summary",
                "summary": last_message.split('\n')[0] if '\n' in last_message else last_message[:200],
                "key_points": [line.strip(' -•*') for line in last_message.split('\n') if line.strip() and not line.startswith(('```', 'Note:', 'sentiment', '---'))][:5] or ["No key points extracted"],
                "sentiment": "neutral",
                "raw_response": last_message,
                "error": str(e),
                "note": "Fallback response due to parsing error"
            }
    
    async def process_url(self, url: str) -> Dict[str, Any]:
        """
        Process a URL: extract content and generate a summary.
        
        Args:
            url (str): The URL to process
            
        Returns:
            Dict[str, Any]: Dictionary containing the extracted content and summary
        """
        print(f"Extracting content from: {url}")
        content = await self.extract_content(url)
        print(f"Content extracted. Length: {len(content)} characters")
        
        print("Generating summary...")
        summary = await self.summarize(content)
        print("Summary generated.")
        
        return {
            "url": url,
            "content_length": len(content),
            "summary": summary
        }


async def main():
    # Example usage
    url = "https://community.jupiter.money/t/your-rewards-experience-got-an-upgrade/55259"
    
    try:
        # Initialize the summarizer
        print("Initializing Ollama summarizer...")
        summarizer = OllamaSummarizer(model_name="llama3")  # or "mistral", "llama2", etc.
        
        # Process the URL
        print(f"\nProcessing URL: {url}")
        result = await summarizer.process_url(url)
        
        # Print results
        print("\n" + "="*80)
        print(f"URL: {result['url']}")
        print(f"Content length: {result['content_length']} characters")
        
        summary = result.get('summary', {})
        
        # Print header
        print("\n" + "="*80)
        print(" SUMMARY ".center(80, '='))
        print("="*80)
        
        # Print title and sentiment
        title = summary.get('title', 'No title provided')
        sentiment = summary.get('sentiment', 'neutral').upper()
        print(f"\n{'Title:':<12} {title}")
        print(f"{'Sentiment:':<12} {sentiment}")
        
        # Print summary
        summary_text = summary.get('summary')
        if summary_text and summary_text != 'N/A':
            print("\n" + "Summary:".ljust(12))
            print("-" * 80)
            print(textwrap.fill(summary_text, width=80))
        
        # Print key points
        key_points = summary.get('key_points', [])
        if key_points and key_points != ["No key points extracted"]:
            print("\nKey Points:")
            print("-" * 80)
            for i, point in enumerate(key_points, 1):
                print(f"  {i}. {point}")
        
        # Print any notes or errors
        if 'note' in summary:
            print("\n" + "!"*80)
            print(f"NOTE: {summary['note']}")
            
        if 'error' in summary:
            print("\n" + "!"*80)
            print("ERROR DETAILS:".center(80))
            print("!"*80)
            print(summary['error'])
        
        # Print the raw response if there was an error or for debugging
        if 'error' in summary or 'raw_response' in summary:
            print("\n" + "-"*80)
            print("RAW RESPONSE (first 500 chars):")
            print("-"*80)
            print(summary.get('raw_response', 'No raw response')[:500] + "..." if len(summary.get('raw_response', '')) > 500 else summary.get('raw_response', 'No raw response'))
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
