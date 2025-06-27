import asyncio
import json
import os
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup

async def extract_policy_content():
    url = "https://jupiter.money/grievance-redressal-policy/"
    result = []
    
    async with AsyncWebCrawler() as crawler:
        # Fetch the webpage content
        response = await crawler.arun(url=url)
        
        # Parse the HTML content
        soup = BeautifulSoup(response.html, 'html.parser')
        
        # Find the main content container - looking for the main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=lambda x: x and 'content' in x.lower())
        
        if not main_content:
            # If still not found, try getting the body
            main_content = soup.find('body')
        
        if not main_content:
            return [{"error": "Could not find main content on the page"}]
        
        # Find all headings and their corresponding content
        current_heading = "Introduction"
        current_content = []
        
        # Get all relevant elements
        elements = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'li'])
        
        for element in elements:
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                # Save the previous section if exists
                if current_heading or current_content:
                    content_text = "\n".join(current_content).strip()
                    if content_text:  # Only add if there's content
                        result.append({
                            "subheading": current_heading,
                            "content": content_text
                        })
                    current_content = []
                
                current_heading = element.get_text(strip=True)
            elif element.name == 'li':
                # Add list items with bullet points
                current_content.append(f"• {element.get_text(strip=True)}")
            elif element.name in ['ul', 'ol']:
                # Process nested lists
                list_items = [f"• {li.get_text(strip=True)}" for li in element.find_all('li', recursive=False)]
                current_content.extend(list_items)
            else:
                # Handle paragraphs
                text = element.get_text(strip=True)
                if text and len(text) > 10:  # Only add non-empty text with reasonable length
                    current_content.append(text)
        
        # Add the last section
        if current_heading or current_content:
            content_text = "\n".join(current_content).strip()
            if content_text:  # Only add if there's content
                result.append({
                    "subheading": current_heading,
                    "content": content_text
                })
        
        return result

async def main():
    # Extract the policy content
    policy_sections = await extract_policy_content()
    
    # Define output file path
    output_dir = "/Users/adityaarya/Documents/jupiter_intern"
    output_file = os.path.join(output_dir, "grievance_policy.json")
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the result to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(policy_sections, f, indent=2, ensure_ascii=False)
    
    print(f"Policy content has been saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())