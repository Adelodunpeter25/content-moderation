"""Data loading service for spam classification datasets."""
import pandas as pd
import requests
from pathlib import Path
from typing import Tuple

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
        
        texts = df['text'].tolist()
        labels = [1 if label == 'spam' else 0 for label in df['label']]
        
        return texts, labels
    
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