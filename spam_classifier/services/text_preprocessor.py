"""Text preprocessing service for spam classification."""
import re
import unicodedata
from typing import str

from core.logging import logger

class TextPreprocessor:
    """Handles text cleaning and normalization for spam classification."""
    
    def __init__(self):
        # Compile regex patterns for better performance
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}')
        self.extra_spaces_pattern = re.compile(r'\s+')
        
    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps to text.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = self._remove_urls(text)
        
        # Remove emails
        text = self._remove_emails(text)
        
        # Remove phone numbers
        text = self._remove_phone_numbers(text)
        
        # Handle emojis and special characters
        text = self._handle_special_characters(text)
        
        # Normalize text
        text = self._normalize_text(text)
        
        return text.strip()
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.url_pattern.sub('[URL]', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self.email_pattern.sub('[EMAIL]', text)
    
    def _remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text."""
        return self.phone_pattern.sub('[PHONE]', text)
    
    def _handle_special_characters(self, text: str) -> str:
        """Handle emojis and special characters."""
        # Convert emojis to text descriptions
        text = self._convert_emojis(text)
        
        # Remove or replace special characters
        text = re.sub(r'[^\w\s\[\]]', ' ', text)
        
        return text
    
    def _convert_emojis(self, text: str) -> str:
        """Convert common emojis to text representations."""
        emoji_map = {
            '😀': 'happy',
            '😂': 'laughing',
            '😍': 'love',
            '🔥': 'fire',
            '💰': 'money',
            '💸': 'money',
            '🎉': 'celebration',
            '🎁': 'gift',
            '⚡': 'lightning',
            '🚨': 'alert'
        }
        
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, f' {replacement} ')
            
        return text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text case and spacing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra spaces
        text = self.extra_spaces_pattern.sub(' ', text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        return text