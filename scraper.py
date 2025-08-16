import os
import re
import json
import time
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from playwright.async_api import async_playwright

# SETTINGS
COSMETICS_URL = "https://fortnite.gg/cosmetics?type=outfit"
BASE_VIDEO_URL = "https://fnggcdn.com/items/{}/video.mp4?1"
OUTPUT_DIR = "fortnite_videos"
METADATA_FILE = "video_metadata.json"
CONCURRENT_DOWNLOADS = 5
RETRY_LIMIT = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Scroll to bottom and extract skin IDs + names
async def extract_skins():
    print("  - Launching browser...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        print("  - Navigating to cosmetics page...")
        try:
            await page.goto(COSMETICS_URL, wait_until="domcontentloaded", timeout=45000)
        except Exception as e:
            print(f"    Error loading page: {e}")
            await browser.close()
            return {}
        
        # Wait for the page to be fully loaded and content to appear
        print("  - Waiting for content to load...")
        try:
            # Wait for any cosmetic items to appear (try multiple selectors)
            await page.wait_for_selector("a[href*='cosmetics'], .item, .card, [data-id]", timeout=15000)
        except:
            print("    Timeout waiting for content, continuing anyway...")
        
        # Additional wait to ensure JS has finished loading
        await page.wait_for_timeout(5000)
        
        # Scroll to bottom until no more new content
        print("  - Scrolling to load all content...")
        last_height = 0
        scroll_count = 0
        while True:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)
            new_height = await page.evaluate("document.body.scrollHeight")
            scroll_count += 1
            print(f"    Scroll #{scroll_count}, height: {new_height}px")
            if new_height == last_height:
                print("    No more content to load.")
                break
            last_height = new_height

        # Extract links and names
        print("  - Extracting skin data...")
        cards = await page.query_selector_all("a.item-icon[href^='/cosmetics?id='][data-id]")
        print(f"    Found {len(cards)} potential skin cards")
        
        skin_data = {}
        for i, card in enumerate(cards):
            if i % 50 == 0 and i > 0:
                print(f"    Processed {i}/{len(cards)} cards...")
                
            href = await card.get_attribute("href")
            title = await card.get_attribute("title") or "unknown"
            match = re.search(r"id=(\d+)", href)
            if match:
                skin_id = match.group(1)
                skin_data[skin_id] = title.strip()

        print(f"  - Successfully extracted {len(skin_data)} skins")
        print("  - Closing browser...")
        await browser.close()
        return skin_data

# Step 2: Download a single video with retry
async def download_video(session, sem, skin_id, skin_name):
    filename = f"{skin_id}_{skin_name.replace(' ', '_')}.mp4"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        return

    url = BASE_VIDEO_URL.format(skin_id)
    retries = 0

    async with sem:
        while retries < RETRY_LIMIT:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(filepath, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        return
                    else:
                        retries += 1
                        await asyncio.sleep(1)
            except Exception:
                retries += 1
                await asyncio.sleep(1)

# Step 3: Download all videos in parallel
async def download_all_videos(skin_data):
    sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
    async with aiohttp.ClientSession() as session:
        tasks = [
            download_video(session, sem, skin_id, skin_data[skin_id])
            for skin_id in skin_data
        ]
        await tqdm.gather(*tasks)

# Step 4: Save metadata
def save_metadata(skin_data):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(skin_data, f, indent=2, ensure_ascii=False)

# Main runner
async def main():
    print("[1] Extracting skin data...")
    skin_data = await extract_skins()
    print(f"[✓] Found {len(skin_data)} skins.")

    print("[2] Saving metadata...")
    save_metadata(skin_data)

    print("[3] Downloading videos...")
    await download_all_videos(skin_data)
    print("[✓] Done!")

if __name__ == "__main__":
    asyncio.run(main())
