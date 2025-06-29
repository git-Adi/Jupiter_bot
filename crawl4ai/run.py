import asyncio
import json
from crawl4ai import *
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

async def main():
    
    url = "https://jupiter.money/grievance-redressal-policy/"
    
    async with AsyncWebCrawler() as crawler:
        print(f"[+] Fetching content from {url}...")
        # Get the content
        result = await crawler.arun(url=url)
        
        print("[+] Generating summary using Llama 3.2 3B...")
        
        try:
            # Generate the summary
            summary = generate_summary(result.markdown)
            
            # Print the result
            print("\n=== Summary ===")
            print(summary)
            
            # Try to parse and pretty-print JSON if possible
            try:
                summary_json = json.loads(summary)
                print("\nFormatted Summary:")
                print(json.dumps(summary_json, indent=2))
            except json.JSONDecodeError:
                print("\nCould not parse as JSON. Raw output:")
                print(summary)
                
        except Exception as e:
            print(f"\nError generating summary: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())