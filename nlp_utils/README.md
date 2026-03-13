# NLP Utilities

This folder contains utilities for natural language processing, including an `NLP_Tokenizer` designed to clean text, stem tokens, and seamlessly hook into Scikit-Learn pipelines.

## Setup Requirements

The utilities in this module rely heavily on the **Natural Language Toolkit (NLTK)**. 

Because NLTK does not package its static word corpora (like stopword lists or the WordNet vocabulary) directly into its PyPI library, they must be downloaded locally.

**Good News:** The `nlp_utils` module handles this for you! 

When you import anything from this module (e.g. `from nlp_utils import NLP_Tokenizer`), the `__init__.py` script automatically verifies if the required corpora (like `stopwords`, `wordnet`, `punkt_tab`, and `averaged_perceptron_tagger_eng`) exist on your machine. If any are missing, it quietly downloads them in the background on its first run.

## Usage Examples

### Option 1: Passing as the `tokenizer` parameter (The Direct Way)

You can pass the `tokenize` method of our custom tokenizer directly into vectorizers that expect a callable tokenizer. Scikit-learn will internally lower-case the text by default before passing it to this method.

```python
from nlp_utils import NLP_Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Initialize your custom tokenizer
my_tokenizer = NLP_Tokenizer(stem=True, min_length=2)

# 2. Pass its `.tokenize` method to the vectorizer
vectorizer = TfidfVectorizer(tokenizer=my_tokenizer.tokenize)

# 3. Fit and transform your raw text!
tfidf_matrix = vectorizer.fit_transform([
    "Raw text document one", 
    "Another raw document!"
])
```

### Option 2: Using the `transform()` method (The Pipeline Way)

Because `NLP_Tokenizer` inherits from `BaseEstimator` and `TransformerMixin`, it functions as a regular Scikit-Learn transformer. Its `transform()` method tokenizes documents and then joins them back together with spaces.

This means you can chain it in a standard `Pipeline` and let the downstream vectorizer use its default space-splitting tokenization! This approach is generally preferred in ML as it makes saving algorithms and data streaming easier over time.

```python
from nlp_utils import NLP_Tokenizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

# Chain them together in a formal ML pipeline
pipe = Pipeline([
    ('text_cleaner', NLP_Tokenizer(stem=True, min_length=2)),
    # CountVectorizer uses default space-tokenization on the already cleaned text
    ('vectorizer', CountVectorizer()) 
])

count_matrix = pipe.fit_transform([
    "Raw text document one", 
    "Another raw document!"
])
```
