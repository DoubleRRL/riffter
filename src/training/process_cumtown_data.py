#!/usr/bin/env python3
"""
Process Cum Town transcript data for fine-tuning a Nick Mullen comedy model
"""

import pandas as pd
import re
import hashlib
from pathlib import Path
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CumTownDataProcessor:
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def load_transcript_csv(self, csv_path: str, delimiter: str = '~') -> pd.DataFrame:
        """Load the Cum Town transcript CSV file"""
        try:
            # Read with custom delimiter
            df = pd.read_csv(csv_path, delimiter=delimiter, header=None, names=['transcript'])
            logger.info(f"Loaded {len(df)} transcript segments from {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def clean_transcript_text(self, text: str) -> str:
        """Clean and normalize transcript text for comedy content"""
        if not isinstance(text, str):
            return ""

        # Remove timestamps and artifacts
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
        text = re.sub(r'\[\d{2}:\d{2}\]', '', text)
        text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause], etc.

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Clean up speaker labels (common in podcasts)
        text = re.sub(r'(Nick|Adam|Joe|Brian|Host|Guest)[\s:]+', r'\1: ', text, flags=re.IGNORECASE)

        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)

        # Remove very short segments (likely artifacts)
        if len(text.split()) < 3:
            return ""

        return text.strip()

    def split_into_chunks(self, text: str, max_words: int = 256) -> List[str]:
        """Split long text into smaller chunks suitable for training"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            if len(chunk.strip()) > 20:  # Only keep substantial chunks
                chunks.append(chunk.strip())

        return chunks

    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """Remove duplicate texts based on content hash"""
        seen_hashes = set()
        unique_texts = []

        for text in texts:
            # Create hash of cleaned text
            text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)

        logger.info(f"Removed {len(texts) - len(unique_texts)} duplicate segments")
        return unique_texts

    def format_for_continued_pretraining(self, texts: List[str]) -> List[str]:
        """Format texts for continued pretraining (simpler approach)"""
        formatted_texts = []

        for text in texts:
            # For continued pretraining, we just need the raw text
            # The model will learn Nick's style from exposure to the transcripts
            formatted_texts.append(text)

        return formatted_texts

    def create_training_dataset(self, csv_path: str) -> pd.DataFrame:
        """Complete pipeline to create training dataset"""
        logger.info("Starting Cum Town data processing pipeline...")

        # Load raw data
        df = self.load_transcript_csv(csv_path)

        # Clean all transcripts
        logger.info("Cleaning transcript text...")
        df['cleaned_text'] = df['transcript'].apply(self.clean_transcript_text)

        # Remove empty entries
        df = df[df['cleaned_text'].str.len() > 0].copy()
        logger.info(f"After cleaning: {len(df)} segments remain")

        # Split long texts into chunks
        logger.info("Splitting long transcripts into chunks...")
        all_chunks = []
        for text in df['cleaned_text']:
            chunks = self.split_into_chunks(text)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} training chunks")

        # Remove duplicates
        logger.info("Removing duplicates...")
        unique_chunks = self.remove_duplicates(all_chunks)
        logger.info(f"After deduplication: {len(unique_chunks)} unique chunks")

        # Format for continued pretraining
        logger.info("Formatting for continued pretraining...")
        formatted_texts = self.format_for_continued_pretraining(unique_chunks)

        # Convert to DataFrame with simple text column
        training_df = pd.DataFrame({"text": formatted_texts})
        logger.info(f"Final training dataset: {len(training_df)} samples")

        return training_df

    def save_training_data(self, df: pd.DataFrame, output_path: str):
        """Save the processed training data"""
        df.to_excel(output_path, index=False)
        logger.info(f"Saved training data to {output_path}")

        # Also save as JSON for flexibility
        json_path = output_path.replace('.xlsx', '.json')
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Also saved as JSON to {json_path}")

def main():
    processor = CumTownDataProcessor()

    # Path to the Cum Town transcripts
    input_csv = "transcripts/ct transcripts.csv"

    if not Path(input_csv).exists():
        logger.error(f"Input CSV not found: {input_csv}")
        logger.error("Please ensure the Cum Town transcripts are in transcripts/ct transcripts.csv")
        return

    # Process the data
    training_df = processor.create_training_dataset(input_csv)

    # Save the results
    output_path = "data/cumtown_training_data.xlsx"
    processor.save_training_data(training_df, output_path)

    logger.info("Cum Town data processing complete!")
    logger.info(f"You can now use {output_path} for training with train.py")

if __name__ == "__main__":
    main()
