import asyncio
import json
import textwrap
import os
from pathlib import Path
from typing import Dict, Any, List
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import ollama
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

class OllamaOnlySummarizer:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.client = ollama.AsyncClient()
    
    async def extract_content(self, url: str) -> Dict[str, Any]:
        async with AsyncWebCrawler() as crawler:
            response = await crawler.arun(url=url)
            soup = BeautifulSoup(response.html, 'html.parser')
            
            # Removing script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Finding all heading elements (h1-h6)
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            result = {
                'url': url,
                'title': soup.title.string if soup.title else 'No Title',
                'sections': []
            }
            
            
            for heading in headings:
                section = {
                    'heading': heading.get_text(strip=True),
                    'level': int(heading.name[1]),  
                    'content': []
                }
                
                # Getting all content until the next heading of same or higher level
                next_node = heading.next_sibling
                while next_node:
                    if next_node.name and next_node.name.startswith('h') and int(next_node.name[1]) <= section['level']:
                        break
                    
                    if next_node.name and next_node.name not in ['script', 'style', 'noscript']:
                        text = next_node.get_text(strip=True)
                        if text:
                            section['content'].append(text)
                    
                    next_node = next_node.next_sibling
                
                result['sections'].append(section)
            
            return result
    
    async def generate_summary(self, content: str) -> Dict[str, Any]: # summary generator
       
        content = content[:12000]
        
        prompt = f"""
        Analyze the following content and provide a structured summary including:
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
        
        try:
            # Generating the summary using Ollama
            response = await self.client.generate(
                model=self.model_name,
                prompt=prompt,
                format="json",
                options={"temperature": 0.3, "top_p": 0.9}
            )
            
            # Extracting and parse the response
            response_text = response['response'].strip()
            
            
            try:
                # Removing markdown code block
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                
                # Parsing the JSON response
                summary = json.loads(response_text)
                
                # Ensuring all required fields are present
                if not all(key in summary for key in ["title", "summary", "key_points", "sentiment"]):
                    raise ValueError("Missing required fields in JSON response")
                
                return summary
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing JSON response: {e}")
                return {
                    "title": "Summary",
                    "summary": response_text.split('\n')[0][:200],
                    "key_points": ["No structured key points could be extracted"],
                    "sentiment": "neutral",
                    "raw_response": response_text,
                    "error": str(e)
                }
                
        except Exception as e:
            print(f"Error generating summary: {e}")
            return {
                "title": "Error",
                "summary": f"Failed to generate summary: {str(e)}",
                "key_points": [],
                "sentiment": "neutral",
                "error": str(e)
            }
    
    async def process_url(self, url: str) -> Dict[str, Any]:
        print(f"Extracting content from: {url}")
        content = await self.extract_content(url)
        print(f"Content extracted. Found {len(content.get('sections', []))} sections")
        
        return content


async def read_urls_from_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

async def process_single_url(summarizer: OllamaOnlySummarizer, url: str) -> Dict[str, Any]:
    try:
        result = await summarizer.process_url(url)
        return {"success": True, "result": result}
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": str(e),
            "result": {
                "url": url,
                "error": str(e),
                "content_length": 0,
                "summary": {
                    "title": "Error",
                    "summary": f"Failed to process URL: {str(e)}",
                    "key_points": [],
                    "sentiment": "error"
                }
            }
        }

async def process_urls(urls: List[str], output_file: str):
    results = []
    summarizer = OllamaOnlySummarizer()
    
    
    pbar = tqdm(
        total=len(urls),
        desc="Processing URLs",
        unit="URL",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        ncols=100
    )
    
    # Processing URLs
    for url in urls:
        try:
            
            result = await summarizer.process_url(url)
            results.append(result)
            
            # Saving 
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            pbar.update(1)
            pbar.set_postfix({"Processed": f"{len(results)}/{len(urls)}"})
            
        except Exception as e:
            error_result = {
                'url': url,
                'error': str(e),
                'sections': []
            }
            results.append(error_result)
            print(f"\n✗ Error processing {url}: {str(e)}")
            
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            pbar.update(1)
    
    pbar.close()
    return results

async def main():
    output_file = "/Users/adityaarya/Documents/jupiter_intern/final_content1.json"
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        urls = [
            'https://jupiter.money/amc-amb-pro-faqs/',
            'https://jupiter.money/grievance-redressal-policy/'
        ]
        
        print(f"Processing {len(urls)} URLs")
        if not urls:
            print("No URLs to process.")
            return
            
        print(f"Found {len(urls)} URLs to process")
        
        print("\nStarting URL processing...")
        results = await process_urls(urls, output_file)
        
        success_count = sum(1 for r in results if 'error' not in r or not r.get('error'))
        error_count = len(urls) - success_count
        total_sections = sum(len(r.get('sections', [])) for r in results)
        
        print("\n" + "="*80)
        print(" PROCESSING SUMMARY ".center(80, '='))
        print("="*80)
        print(f"\n{'Total URLs:':<25} {len(urls)}")
        print(f"{'Successfully processed:':<25} \033[92m{success_count}\033[0m")
        print(f"{'Failed:':<25} \033[91m{error_count}\033[0m")
        print(f"{'Total sections extracted:':<25} {total_sections}")
        print(f"\nResults saved to: \033[94m{output_file}\033[0m")
        
        if error_count > 0:
            print("\nFailed URLs:")
            for result in results:
                if 'error' in result and result['error']:
                    print(f"- {result['url']}: {result['error']}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
