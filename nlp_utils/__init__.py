import nltk

def _ensure_nltk_data():
    """
    Ensures that the required NLTK corpora are downloaded.
    If they are missing, it quietly downloads them upon the first import.
    """
    required_corpora = [
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    ]
    
    for resource_path, download_name in required_corpora:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)

# Run the check when the package is initialized
_ensure_nltk_data()

# Expose the tokenizer to the package level
from .nlp_tokenizer import NLP_Tokenizer
