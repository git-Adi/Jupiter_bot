# import asyncio
# import json
# from crawl4ai import *
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Load Llama 3.2 3B Instruct model and tokenizer
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# print(f"Loading {model_name}...")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     torch_dtype="auto"
# )

# # Create a text generation pipeline
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.9,
# )

# def generate_summary(text: str) -> str:
#     """Generate a summary using the Llama model."""
#     prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#     You are a helpful assistant that summarizes forum discussions on the basis of what aspects of Jupiter Money are being discussed. 
#     Provide a JSON response with 'title' and 'summary' fields.
    
#     Forum content:
#     {text[:4000]}
    
#     Please analyze this forum discussion and provide:
#     1. A concise title (3-10 words)
#     2. A brief summary (4-10 sentences)
    
#     Format your response as a JSON object with 'title' and 'summary' fields.
#     <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""
    
#     # Generate response
#     response = pipe(prompt)[0]['generated_text']
    
#     # Extract only the assistant's response
#     assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[-1].strip()
#     return assistant_response

# async def main():
    
#     url = "https://community.jupiter.money/t/process-for-service-requests-please-follow-this-important/14910"
    
#     async with AsyncWebCrawler() as crawler:
#         print(f"[+] Fetching content from {url}...")
#         # Get the content
#         result = await crawler.arun(url=url)
        
#         print("[+] Generating summary using Llama 3.2 3B...")
        
#         try:
#             # Generate the summary
#             summary = generate_summary(result.markdown)
            
#             # Print the result
#             print("\n=== Summary ===")
#             print(summary)
            
#             # Try to parse and pretty-print JSON if possible
#             try:
#                 summary_json = json.loads(summary)
#                 print("\nFormatted Summary:")
#                 print(json.dumps(summary_json, indent=2))
#             except json.JSONDecodeError:
#                 print("\nCould not parse as JSON. Raw output:")
#                 print(summary)
                
#         except Exception as e:
#             print(f"\nError generating summary: {str(e)}")

# if __name__ == "__main__":
#     asyncio.run(main())

# import asyncio
# import os
# from tqdm import tqdm
# from crawl4ai import *
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Load Llama 3.2 3B Instruct model and tokenizer
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# print(f"Loading {model_name}...")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     torch_dtype="auto"
# )

# # Create a text generation pipeline
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.9,
# )

# def generate_summary(text: str) -> str:
#     """Generate a summary using the Llama model."""
#     prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#     You are a helpful assistant that summarizes forum discussions on the basis of what aspects of Jupiter Money are being discussed. 
#     Provide a JSON response with 'title' and 'summary' fields.
    
#     Forum content:
#     {text[:4000]}
    
#     Please analyze this forum discussion and provide:
#     1. A concise title (3-10 words)
#     2. A brief summary (4-10 sentences)
    
#     Format your response as a JSON object with 'title' and 'summary' fields.
#     <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""
    
#     # Generate response
#     response = pipe(prompt)[0]['generated_text']
    
#     # Extract only the assistant's response
#     assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[-1].strip()
#     return assistant_response

# def read_urls_from_file(file_path: str) -> list:
#     """Read URLs from a text file and return a list of valid forum URLs."""
#     with open(file_path, 'r') as f:
#         # Filter only Jupiter community forum URLs and remove any quotes or whitespace
#         urls = [line.strip().strip('"\'') for line in f if 'community.jupiter.money/t/' in line]
#     return urls

# async def process_url(crawler, url: str, output_file: str):
#     """Process a single URL and append the result to the output file."""
#     try:
#         # Get the content
#         result = await crawler.arun(url=url)
        
#         # Generate the summary
#         summary = generate_summary(result.markdown)
        
#         # Append to the single output file
#         with open(output_file, 'a', encoding='utf-8') as f:
#             f.write(f"URL: {url}\n")
#             f.write(f"Summary: {summary}\n")
#             f.write("-" * 80 + "\n\n")
        
#         return f"Processed: {url}"
#     except Exception as e:
#         return f"Error processing {url}: {str(e)}"

# async def main():
#     # Read URLs from file
#     urls = read_urls_from_file("/Users/adityaarya/Documents/jupiter_intern/extracted_urls.txt")
#     output_file = "/Users/adityaarya/Documents/jupiter_intern/forum_summaries.txt"
    
#     print(f"Found {len(urls)} forum URLs to process")
#     print(f"Output will be saved to: {output_file}")
    
#     # Clear the output file if it exists
#     if os.path.exists(output_file):
#         os.remove(output_file)
    
#     async with AsyncWebCrawler() as crawler:
#         # Process URLs with progress bar
#         for url in tqdm(urls, desc="Processing URLs"):
#             result = await process_url(crawler, url, output_file)
#             tqdm.write(result)  # Write progress to console without interfering with the progress bar

# if __name__ == "__main__":
#     asyncio.run(main())



import asyncio
import os
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from crawl4ai import *
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Global variables for model and pipeline
model = None
pipe = None

def initialize_model():
    """Initialize the model and pipeline."""
    global model, pipe
    # Load Llama 3.2 3B Instruct model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    # Create a text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

async def generate_summary_async(text: str, executor: ThreadPoolExecutor) -> str:
    """Run the summary generation in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_summary, text)

def generate_summary(text: str) -> str:
    """Generate a summary using the Llama model."""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant that summarizes forum discussions on the basis of what aspects of Jupiter Money are being discussed. 
    Provide a JSON response with 'title' and 'summary' fields.
    
    Forum content:
    {text[:4000]}
    
    Please analyze this forum discussion and provide:
    1. A concise title (3-10 words)
    2. A brief summary (4-10 sentences)
    
    Format your response as a JSON object with 'title' and 'summary' fields.
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""
    
    # Generate response
    response = pipe(prompt)[0]['generated_text']
    
    # Extract only the assistant's response
    assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>\n")[-1].strip()
    return assistant_response

def read_urls_from_file(file_path: str) -> list:
    """Read URLs from a text file and return a list of valid forum URLs."""
    with open(file_path, 'r') as f:
        # Filter only Jupiter community forum URLs and remove any quotes or whitespace
        urls = [line.strip().strip('"\'') for line in f if 'community.jupiter.money/t/' in line]
    return urls

async def process_url(crawler, url: str, output_file: str, executor: ThreadPoolExecutor, semaphore: asyncio.Semaphore):
    """Process a single URL and append the result to the output file."""
    async with semaphore:  # This limits concurrent executions
        try:
            # Get the content
            result = await crawler.arun(url=url)
            
            # Generate the summary in a thread
            summary = await generate_summary_async(result.markdown, executor)
            
            # Use aiofiles for thread-safe file writing
            async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
                await f.write(f"URL: {url}\n")
                await f.write(f"Summary: {summary}\n")
                await f.write("-" * 80 + "\n\n")
            
            return f"Processed: {url}"
        except Exception as e:
            return f"Error processing {url}: {str(e)}"

async def process_batch(urls: list, output_file: str, max_concurrent: int = 5):
    """Process a batch of URLs with controlled concurrency."""
    # Initialize the model once
    initialize_model()
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create a thread pool for CPU-bound tasks
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        async with AsyncWebCrawler() as crawler:
            # Create tasks for all URLs
            tasks = [process_url(crawler, url, output_file, executor, semaphore) 
                    for url in urls]
            
            # Process with progress bar
            results = []
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing URLs"):
                result = await f
                if result:
                    tqdm.write(result)  # Write progress to console
                    results.append(result)
            
            return results

async def main():
    # Read URLs from file
    input_file = "/Users/adityaarya/Documents/jupiter_intern/extracted_urls.txt"
    output_file = "/Users/adityaarya/Documents/jupiter_intern/forum_summaries.txt"
    
    urls = read_urls_from_file(input_file)
    
    print(f"Found {len(urls)} forum URLs to process")
    print(f"Output will be saved to: {output_file}")
    
    # Clear the output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Process URLs with controlled concurrency
    await process_batch(urls, output_file, max_concurrent=5)

if __name__ == "__main__":
    asyncio.run(main())