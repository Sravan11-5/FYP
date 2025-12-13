"""
English to Telugu Translation Utility
Uses Google Translate via deep-translator
"""
from deep_translator import GoogleTranslator
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class TeluguTranslator:
    """Handles English to Telugu translation"""
    
    def __init__(self):
        self.translator = GoogleTranslator(source='en', target='te')
    
    def translate_text(self, text: str) -> Optional[str]:
        """
        Translate English text to Telugu
        
        Args:
            text: English text to translate
            
        Returns:
            Telugu translation or None if failed
        """
        if not text or not text.strip():
            return None
            
        try:
            # Google Translate has 5000 char limit, truncate if needed
            text = text[:4999] if len(text) > 5000 else text
            translated = self.translator.translate(text)
            logger.info(f"Translated: {text[:50]}... -> {translated[:50]}...")
            return translated
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return None
    
    def translate_batch(self, texts: List[str], max_retries: int = 2) -> List[str]:
        """
        Translate multiple English texts to Telugu
        
        Args:
            texts: List of English texts
            max_retries: Number of retry attempts for failed translations
            
        Returns:
            List of Telugu translations (empty string if translation failed)
        """
        translated = []
        
        for text in texts:
            if not text or not text.strip():
                translated.append("")
                continue
            
            # Try translating with retries
            for attempt in range(max_retries + 1):
                try:
                    result = self.translate_text(text)
                    if result:
                        translated.append(result)
                        break
                    else:
                        if attempt == max_retries:
                            logger.warning(f"Failed to translate after {max_retries} attempts: {text[:50]}...")
                            translated.append("")  # Use empty string as fallback
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Translation error after {max_retries} attempts: {e}")
                        translated.append("")  # Use empty string as fallback
        
        return translated

# Singleton instance
_translator_instance = None

def get_translator() -> TeluguTranslator:
    """Get singleton translator instance"""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = TeluguTranslator()
    return _translator_instance
