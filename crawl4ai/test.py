import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://community.jupiter.money/t/your-rewards-experience-got-an-upgrade/55259",
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())