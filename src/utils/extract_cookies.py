#!/usr/bin/env python3
"""
Extract YouTube cookies from Safari for yt-dlp
"""

import sqlite3
import os
import json
from pathlib import Path
import http.cookiejar
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_safari_cookies():
    """Extract YouTube cookies from Safari's cookie storage"""
    try:
        # Safari cookies location
        cookie_path = Path.home() / "Library/Containers/com.apple.Safari/Data/Library/Cookies/Cookies.binarycookies"

        if not cookie_path.exists():
            logger.error(f"Safari cookies not found at: {cookie_path}")
            return None

        logger.info("Found Safari cookies file")

        # For now, let's create a manual approach
        logger.info("Safari cookies are in binary format and require special extraction")
        logger.info("Please try one of these alternatives:")
        logger.info("1. Use Chrome/Firefox with yt-dlp --cookies-from-browser chrome")
        logger.info("2. Manually download the video using your browser and place it in the audio/ folder")
        logger.info("3. Export cookies manually from Safari")

        return None

    except Exception as e:
        logger.error(f"Failed to extract cookies: {e}")
        return None

def create_manual_cookie_file():
    """Create a template for manual cookie entry"""
    template = """# YouTube Cookies Template
# Copy this format and replace with your actual YouTube cookies from Safari
# You can get these by:
# 1. Open Safari with YouTube logged in
# 2. Open Developer Tools (Safari > Develop > Show Web Inspector)
# 3. Go to Storage > Cookies > youtube.com
# 4. Copy the relevant cookies (especially those with __Secure- prefix)

[
    {
        "domain": ".youtube.com",
        "expirationDate": 1735689600,
        "hostOnly": false,
        "httpOnly": true,
        "name": "VISITOR_INFO1_LIVE",
        "path": "/",
        "sameSite": "no_restriction",
        "secure": true,
        "session": false,
        "storeId": "0",
        "value": "YOUR_COOKIE_VALUE_HERE"
    }
]
"""

    cookie_file = Path("youtube_cookies.json")
    with open(cookie_file, 'w') as f:
        f.write(template)

    logger.info(f"Created cookie template at: {cookie_file}")
    logger.info("Edit this file with your actual YouTube cookies from Safari")
    return str(cookie_file)

def test_yt_dlp_with_cookies():
    """Test yt-dlp with the cookie file"""
    cookie_file = Path("youtube_cookies.json")

    if not cookie_file.exists():
        logger.error("Cookie file not found. Run create_manual_cookie_file() first")
        return False

    try:
        import subprocess
        yt_dlp_path = os.path.join("venv", "bin", "yt-dlp")
        cmd = [
            yt_dlp_path,
            "--cookies", str(cookie_file),
            "--print", "%(title)s",
            "https://www.youtube.com/watch?v=yzJZtaWFB_s"
        ]

        logger.info("Testing yt-dlp with cookies...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())

        if result.returncode == 0:
            logger.info("Cookie test successful!")
            logger.info(f"Video title: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"Cookie test failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Cookie test error: {e}")
        return False

if __name__ == "__main__":
    logger.info("YouTube Cookie Extraction Tool")

    # Try automatic extraction
    cookies = extract_safari_cookies()

    if cookies:
        logger.info("Cookies extracted successfully!")
    else:
        logger.info("Automatic extraction failed, creating manual template...")
        create_manual_cookie_file()

        logger.info("\nNext steps:")
        logger.info("1. Edit youtube_cookies.json with your actual cookies")
        logger.info("2. Run: python extract_cookies.py test")
        logger.info("3. If successful, use --cookies youtube_cookies.json with yt-dlp")
