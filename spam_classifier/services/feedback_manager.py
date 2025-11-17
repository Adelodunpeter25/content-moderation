"""Feedback management for spam classification improvements."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from core.logging import logger

class FeedbackManager:
    """Manages user feedback for model improvement."""
    
    def __init__(self):
        self.feedback_dir = Path("data/feedback")
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "user_feedback.jsonl"
        
    def record_feedback(self, text: str, predicted_spam: bool, actual_spam: bool, 
                       confidence: float, user_id: str = "anonymous") -> None:
        """Record user feedback about prediction accuracy.
        
        Args:
            text: The text that was classified
            predicted_spam: What the model predicted
            actual_spam: What the user says it actually is
            confidence: Model's confidence in prediction
            user_id: ID of user providing feedback
        """
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "predicted_spam": predicted_spam,
            "actual_spam": actual_spam,
            "confidence": confidence,
            "user_id": user_id,
            "is_correction": predicted_spam != actual_spam
        }
        
        # Append to JSONL file
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
            
        logger.info(f"Recorded feedback: {'correction' if feedback_entry['is_correction'] else 'confirmation'}")
    
    def get_corrections(self, limit: int = 100) -> List[Dict]:
        """Get recent corrections for retraining.
        
        Args:
            limit: Maximum number of corrections to return
            
        Returns:
            List of correction entries
        """
        if not self.feedback_file.exists():
            return []
            
        corrections = []
        with open(self.feedback_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry['is_correction']:
                    corrections.append(entry)
                    
        # Return most recent corrections
        return corrections[-limit:] if corrections else []
    
    def get_training_data_from_feedback(self) -> Tuple[List[str], List[int]]:
        """Extract training data from user feedback.
        
        Returns:
            Tuple of (texts, labels) from user corrections
        """
        corrections = self.get_corrections()
        
        texts = [entry['text'] for entry in corrections]
        labels = [1 if entry['actual_spam'] else 0 for entry in corrections]
        
        logger.info(f"Extracted {len(texts)} training samples from user feedback")
        return texts, labels
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics about user feedback.
        
        Returns:
            Dictionary with feedback statistics
        """
        if not self.feedback_file.exists():
            return {"total": 0, "corrections": 0, "accuracy": 0.0}
            
        total = 0
        corrections = 0
        
        with open(self.feedback_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                total += 1
                if entry['is_correction']:
                    corrections += 1
        
        accuracy = ((total - corrections) / total * 100) if total > 0 else 0.0
        
        return {
            "total_feedback": total,
            "corrections": corrections,
            "accuracy_percent": round(accuracy, 2)
        }