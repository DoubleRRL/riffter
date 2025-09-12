#!/usr/bin/env python3
"""
YouTube Transcript Extractor for Cum Town and Nick Mullen content
Extracts transcripts from YouTube videos and playlists, removes duplicates, and exports to XLSX
"""

import pandas as pd
import json
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
from urllib.parse import parse_qs, urlparse

# YouTube API imports
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeTranscriptExtractor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.youtube = None
        if api_key:
            self.youtube = build('youtube', 'v3', developerKey=api_key)

        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.formatter = TextFormatter()

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        parsed_url = urlparse(url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path.startswith('/embed/'):
                return parsed_url.path.split('/embed/')[1]
            elif parsed_url.path.startswith('/v/'):
                return parsed_url.path.split('/v/')[1]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        return url  # Assume it's already a video ID

    def extract_playlist_id(self, url: str) -> str:
        """Extract playlist ID from YouTube playlist URL"""
        parsed_url = urlparse(url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            return parse_qs(parsed_url.query)['list'][0]
        return url  # Assume it's already a playlist ID

    def get_playlist_videos(self, playlist_url: str) -> List[Dict]:
        """Get all video IDs and titles from a YouTube playlist"""
        if not self.youtube:
            logger.warning("YouTube API key not provided - attempting basic playlist parsing")
            # For now, return empty list and handle individual videos
            # User will need to provide individual video URLs from the playlist
            logger.info("Please provide individual video URLs from the playlist for extraction")
            return []

        try:
            playlist_id = self.extract_playlist_id(playlist_url)
            videos = []
            next_page_token = None

            while True:
                request = self.youtube.playlistItems().list(
                    part='snippet',
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                )
                response = request.execute()

                for item in response['items']:
                    video_id = item['snippet']['resourceId']['videoId']
                    title = item['snippet']['title']
                    videos.append({
                        'video_id': video_id,
                        'title': title,
                        'url': f'https://www.youtube.com/watch?v={video_id}'
                    })

                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break

                time.sleep(0.1)  # Rate limiting

            logger.info(f"Found {len(videos)} videos in playlist")
            return videos

        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting playlist videos: {e}")
            return []

    def get_transcript(self, video_id: str, title: str = "") -> Optional[Dict]:
        """Get transcript for a single YouTube video"""
        try:
            # Create API instance and get transcript list
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)

            # Get the first available transcript (prefer manual over auto-generated)
            transcript = None
            try:
                # Try to find manually created transcript first
                transcript = transcript_list.find_manually_created_transcript(['en'])
            except:
                try:
                    # Fall back to auto-generated
                    transcript = transcript_list.find_generated_transcript(['en'])
                except:
                    # Use first available transcript
                    if len(transcript_list) > 0:
                        transcript = transcript_list[0]

            if not transcript:
                logger.warning(f"No transcript available for video {video_id}")
                return None

            # Fetch the actual transcript data
            transcript_data = transcript.fetch()

            # Format as text
            text = self.formatter.format_transcript(transcript_data)

            # Create transcript data
            transcript_info = {
                'video_id': video_id,
                'title': title,
                'url': f'https://www.youtube.com/watch?v={video_id}',
                'transcript': text,
                'duration': sum(item['duration'] for item in transcript_data),
                'word_count': len(text.split()),
                'hash': hashlib.md5(text.encode()).hexdigest()
            }

            logger.info(f"Successfully extracted transcript for: {title[:50]}...")
            return transcript_info

        except Exception as e:
            logger.warning(f"Could not get transcript for video {video_id}: {e}")
            return None

    def clean_transcript_text(self, text: str) -> str:
        """Clean and normalize transcript text"""
        # Remove timestamps and formatting artifacts
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
        text = re.sub(r'\[\d{2}:\d{2}\]', '', text)

        # Remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)

        # Remove common YouTube artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause], etc.
        text = re.sub(r'\(.*?\)', '', text)  # Remove (laughs), etc.

        # Clean up speaker labels for comedy content
        text = re.sub(r'(Nick|Adam|Joe|Brian|Host|Guest)[\s:]+', r'\1: ', text, flags=re.IGNORECASE)

        return text.strip()

    def find_duplicates(self, transcripts: List[Dict]) -> List[Dict]:
        """Find and remove duplicate transcripts based on content similarity"""
        if not transcripts:
            return []

        # Sort by hash first (exact duplicates)
        seen_hashes = set()
        unique_transcripts = []

        for transcript in transcripts:
            transcript_hash = transcript['hash']
            if transcript_hash not in seen_hashes:
                seen_hashes.add(transcript_hash)
                unique_transcripts.append(transcript)

        logger.info(f"Removed {len(transcripts) - len(unique_transcripts)} exact duplicates")

        # Additional similarity-based deduplication could be added here
        # For now, we'll stick with hash-based deduplication

        return unique_transcripts

    def extract_from_video(self, video_url: str) -> Optional[Dict]:
        """Extract transcript from a single video"""
        video_id = self.extract_video_id(video_url)
        return self.get_transcript(video_id, f"Video_{video_id}")

    def extract_from_playlist(self, playlist_url: str) -> List[Dict]:
        """Extract transcripts from all videos in a playlist"""
        videos = self.get_playlist_videos(playlist_url)
        transcripts = []

        for i, video in enumerate(videos):
            logger.info(f"Processing video {i+1}/{len(videos)}: {video['title'][:50]}...")

            transcript = self.get_transcript(video['video_id'], video['title'])
            if transcript:
                # Clean the transcript text
                transcript['transcript'] = self.clean_transcript_text(transcript['transcript'])
                transcripts.append(transcript)

            # Rate limiting
            time.sleep(1)

        return transcripts

    def save_to_json(self, transcripts: List[Dict], filename: str = "transcripts.json"):
        """Save transcripts to JSON file"""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(transcripts, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(transcripts)} transcripts to {filepath}")

    def save_to_xlsx(self, transcripts: List[Dict], filename: str = "transcripts.xlsx"):
        """Save transcripts to Excel file for fine-tuning"""
        if not transcripts:
            logger.warning("No transcripts to save")
            return

        # Prepare data for Excel
        data = []
        for transcript in transcripts:
            data.append({
                'Video ID': transcript['video_id'],
                'Title': transcript['title'],
                'URL': transcript['url'],
                'Duration (seconds)': transcript['duration'],
                'Word Count': transcript['word_count'],
                'Transcript': transcript['transcript'],
                'Hash': transcript['hash']
            })

        df = pd.DataFrame(data)

        # Save to Excel
        filepath = self.data_dir / filename
        df.to_excel(filepath, index=False, engine='openpyxl')
        logger.info(f"Saved {len(transcripts)} transcripts to {filepath}")

    def process_urls(self, urls: List[str]) -> List[Dict]:
        """Process multiple YouTube URLs (videos and playlists)"""
        all_transcripts = []

        for url in urls:
            logger.info(f"Processing URL: {url}")

            if 'playlist' in url.lower():
                # It's a playlist
                transcripts = self.extract_from_playlist(url)
                all_transcripts.extend(transcripts)
            else:
                # It's a single video
                transcript = self.extract_from_video(url)
                if transcript:
                    transcript['transcript'] = self.clean_transcript_text(transcript['transcript'])
                    all_transcripts.append(transcript)

        # Remove duplicates
        unique_transcripts = self.find_duplicates(all_transcripts)

        logger.info(f"Total transcripts collected: {len(all_transcripts)}")
        logger.info(f"Unique transcripts after deduplication: {len(unique_transcripts)}")

        return unique_transcripts


def main():
    # URLs provided by user
    urls = [
        "https://www.youtube.com/watch?v=yzJZtaWFB_s",  # Nick Mullen impressions video
        "https://www.youtube.com/playlist?list=PLzSzmoNOUMVMN6-yJQAElaaMbp7DC9BaO"  # Cum Town compilations playlist
    ]

    # Initialize extractor (without API key for basic transcript extraction)
    extractor = YouTubeTranscriptExtractor()

    # Process all URLs
    transcripts = extractor.process_urls(urls)

    # Save results
    if transcripts:
        extractor.save_to_json(transcripts, "youtube_transcripts.json")
        extractor.save_to_xlsx(transcripts, "youtube_transcripts.xlsx")
        logger.info("Extraction complete!")
    else:
        logger.warning("No transcripts were successfully extracted")


if __name__ == "__main__":
    main()
