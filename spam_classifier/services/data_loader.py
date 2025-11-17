"""Data loading service for spam classification datasets."""
import pandas as pd
import requests
from pathlib import Path
from typing import Tuple

from core.logging import logger

class DataLoader:
    """Loads and prepares spam classification datasets."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
    
    def load_sms_spam_dataset(self) -> Tuple[list[str], list[int]]:
        """Load SMS Spam Collection dataset.
        
        Returns:
            Tuple of (texts, labels) where labels are 1 for spam, 0 for ham
        """
        file_path = self.data_dir / "sms_spam.csv"
        
        if not file_path.exists():
            self._download_sms_dataset(file_path)
        
        df = pd.read_csv(file_path, encoding='latin-1')
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
        
        texts = [str(text) for text in df['text'].tolist()]
        labels = [1 if label == 'spam' else 0 for label in df['label']]
        
        return texts, labels
    
    def load_enron_spam_dataset(self) -> Tuple[list[str], list[int]]:
        """Load Enron email spam dataset.
        
        Returns:
            Tuple of (texts, labels) where labels are 1 for spam, 0 for ham
        """
        # Placeholder for Enron dataset - requires more complex processing
        print("Enron dataset not implemented yet")
        return self.load_sms_spam_dataset()
    
    def load_youtube_spam_dataset(self) -> Tuple[list[str], list[int]]:
        """Load YouTube spam comments dataset.
        
        Returns:
            Tuple of (texts, labels) where labels are 1 for spam, 0 for ham
        """
        file_path = self.data_dir / "youtube_spam.csv"
        
        if not file_path.exists():
            self._download_youtube_dataset(file_path)
        
        df = pd.read_csv(file_path, encoding='latin-1')
        texts = [str(text) for text in df['CONTENT'].tolist()]
        labels = df['CLASS'].tolist()
        
        return texts, labels
    

    
    def load_combined_datasets(self) -> Tuple[list[str], list[int]]:
        """Load and combine multiple spam datasets.
        
        Returns:
            Tuple of (texts, labels) where labels are 1 for spam, 0 for ham
        """
        logger.info("Loading combined spam datasets...")
        
        # Load SMS dataset
        sms_texts, sms_labels = self.load_sms_spam_dataset()
        logger.info(f"Loaded {len(sms_texts)} SMS samples")
        
        # Load YouTube dataset
        youtube_texts, youtube_labels = self.load_youtube_spam_dataset()
        logger.info(f"Loaded {len(youtube_texts)} YouTube samples")
        
        # Combine datasets
        combined_texts = sms_texts + youtube_texts
        combined_labels = sms_labels + youtube_labels
        
        logger.info(f"Total combined samples: {len(combined_texts)}")
        return combined_texts, combined_labels
    
    def _download_sms_dataset(self, file_path: Path) -> None:
        """Download SMS Spam Collection dataset."""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        
        print("Downloading SMS Spam Collection dataset...")
        response = requests.get(url)
        
        import zipfile
        import io
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            with zip_file.open('SMSSpamCollection') as spam_file:
                content = spam_file.read().decode('utf-8')
                
                # Parse the tab-separated format
                lines = content.strip().split('\n')
                data = []
                for line in lines:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        data.append({'v1': parts[0], 'v2': parts[1]})
                
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
        
        print(f"Dataset saved to {file_path}")
    
    def _download_youtube_dataset(self, file_path: Path) -> None:
        """Download YouTube spam comments dataset."""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"
        
        print("Downloading YouTube spam dataset...")
        response = requests.get(url)
        
        import zipfile
        import io
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Combine all CSV files in the zip
            all_data = []
            for file_name in zip_file.namelist():
                if file_name.endswith('.csv'):
                    with zip_file.open(file_name) as csv_file:
                        df = pd.read_csv(csv_file, encoding='latin-1')
                        all_data.append(df)
            
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.to_csv(file_path, index=False)
        
        print(f"YouTube dataset saved to {file_path}")