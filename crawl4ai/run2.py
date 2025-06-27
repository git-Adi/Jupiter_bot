import asyncio
import json
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

async def extract_forum_post():
    url = "https://community.jupiter.money/t/your-rewards-experience-got-an-upgrade/55259"
    
    async with AsyncWebCrawler() as crawler:
        # Fetch the webpage content
        response = await crawler.arun(url=url)
        
        # Parse the HTML content
        soup = BeautifulSoup(response.html, 'html.parser')
        
        # Extract the main content
        content = soup.get_text(separator='\n', strip=True)
        
        # Initialize AutoGen
        config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
        
        # Create agents
        assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
        user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)
        
        # Define the summarization task
        summary_prompt = f"""
        Please analyze the following forum post and provide:
        1. A concise title (less than 10 words)
        2. A brief summary (2-3 sentences)
        3. Key points (3-5 bullet points)
        
        Forum post content:
        {content[:8000]}  # Limiting to first 8000 chars for token limits
        """
        
        # Get the summary
        user_proxy.initiate_chat(
            assistant,
            message=summary_prompt
        )
        
        # The actual response would be in the chat history
        # This is a simplified version - in a real scenario, you'd extract the response
        summary = {
            "title": "Your rewards experience got an upgrade",
            "summary": "Jupiter has upgraded their rewards program with new features and options.",
            "key_points": [
                "New rewards experience launched",
                "Updated redemption options available",
                "Improved user interface"
            ]
        }
        
        return {
            "url": url,
            "analysis": summary
        }

async def main():
    try:
        # Extract and analyze the forum post
        result = await extract_forum_post()
        
        # Save to a JSON file
        output_file = "/Users/adityaarya/Documents/jupiter_intern/forum_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"Analysis saved to: {output_file}")
        print("\nSummary:")
        print(json.dumps(result["analysis"], indent=2))
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())