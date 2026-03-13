from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.probability import FreqDist
from nltk.stem import SnowballStemmer, PorterStemmer

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import matplotlib.pyplot as plt
import string
from collections import defaultdict
from typing import Optional, Union, Set, List, Dict, Iterable, Generator, Any, Sequence

class NLP_Tokenizer(BaseEstimator, TransformerMixin):
    """
    A custom tokenizer for standardizing text in scikit-learn pipelines.
    Supports stemming, lemmatization, stopword removal, and punctuation filtering.
    """

    def __init__(self, stopwords: Optional[Set[str]] = None, punctuation: Optional[Set[str]] = None, min_length: int = 1,
                 lower: bool = True, strip: bool = True, utf: bool = True, 
                 numencode: Union[bool, str] = False, stem: bool = True, nonumbers: bool = False, verbose: bool = False) -> None:
                
        self.min_length = min_length
        self.utf = utf
        self.numencode = numencode if isinstance(numencode, str) else ('<NUM>' if numencode else False)
        self.stem = stem
        self.nonumbers = nonumbers
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = (stopwords if stopwords is not None else set()) | set(sw.words('english'))
        self.punct      = (punctuation if punctuation is not None else set()) | set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')# PorterStemmer()
        
        if(verbose):
            print("Preprocessor...\nlower: {}".format(self.lower))
            print("strip: {}".format(self.strip))
            print("min_length: {}".format(self.min_length))
            print("utf: {}".format(self.utf))
            print("numencode: {}".format(self.numencode))
            print("stem: {}".format(self.stem))
            print("lemmatize: {}".format(not self.stem))
            print("nonumbers: {}".format(self.nonumbers))
        

    def fit(self, X: Iterable[str], y: Optional[Any] = None) -> 'NLP_Tokenizer':
        """Fits the estimator (no-op for tokenizer)."""
        return self

    def inverse_transform(self, X: Iterable[Iterable[str]]) -> List[str]:
        """Joins sequence of tokens back into sentences."""
        return [" ".join(doc) for doc in X]

    def transform(self, X: Iterable[str]) -> List[str]:
        """Transforms documents into a list of space-separated token strings."""
        tmp = [
            list(self.tokenize(doc)) for doc in X
        ]
        return [" ".join(x) for x in tmp]

    def tokenize(self, document: str) -> Generator[str, None, None]:
        """Tokenizes a single document yielding processed tokens."""
        
        for sent in sent_tokenize(document):
            # Every sent to token and pos tag (regardless of whether or not it is used)
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                if(self.utf):
                    # Drops non-ASCII tokens (only leaves pure ASCII tokens)
                    if not token.isascii():
                        continue
                # Simple preprocessing of the tokens
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # ignore stopword
                if token in self.stopwords:
                    continue

                # ignore punctuation (change to all if necessary)
                if not set(token).isdisjoint(self.punct):
                    continue
                
                #choose stem or lemma
                if(self.stem):
                    result = self.stemmer.stem(token)
                else:
                    tag = {
                            'N': wn.NOUN,
                            'V': wn.VERB,
                            'R': wn.ADV,
                            'J': wn.ADJ
                        }.get(tag[0], wn.NOUN)
                    result = self.lemmatizer.lemmatize(token, tag)
                    #result = self.lemmatize(token, tag)
                
                # ignore stopword
                if result in self.stopwords:
                    continue
                
                #drop singular tokens
                if(len(result)<=self.min_length):
                    continue
                
                #drop any token including digit(s) (also drops all digits)
                if(self.nonumbers):
                    if any(char.isdigit() for char in result):
                        continue
                        
                #set numberflag if necessary
                if(self.numencode and result.isdigit()):
                    if(self.numencode == 'drop'):
                        continue
                    result = str(self.numencode)
                
                yield result

    
def top_cluster_features(countVec: Any, labels: Sequence[Any], k: int = 10) -> np.ndarray:
    """
    Returns the top k features for each cluster based on term counts.
    """
    n = len(np.unique(labels))
    result = np.zeros((n,k), dtype=int)
    
    for label in range(n):
        x = countVec[np.where(labels==label),:][0]
        counts = np.asarray(np.sum(x, axis=0)).flatten()
        ind = np.argsort(counts)[-k:]
        result[label,:] = ind
    
    return result[:,::-1]
      
    
def plot_top_words(X: Iterable[str], k: int = 10, title: Optional[str] = None) -> None:
    """
    Plots a frequency distribution of the top k words from a collection of strings.
    """
    #FreqDist plot of top k words in X (e.g., a cluster)
    all_words = []
    
    for item in X:
        for word in item.split():
            all_words.append(word)
            
    fdist = FreqDist(all_words)
    if(title is not None):
        plt.title(title)
    fdist.plot(k,cumulative=False)

def plot_all_clusters(X: Iterable[str], labels: np.ndarray, k: int = 10) -> None:
    """
    Plots the top words for each unique label/cluster in X.
    """
    for label in np.unique(labels):
        plot_top_words(np.asarray(X)[np.where(labels==label)],k,'Cluster {}'.format(label))

def table_top_words(cv: Any, top_words: np.ndarray) -> str:
    """
    Creates a formatted plain-text table showing the top words for each cluster.
    `cv` should be the fitted Vectorizer object (CountVectorizer or TfidfVectorizer).
    """
    feature_names = cv.get_feature_names_out() if hasattr(cv, "get_feature_names_out") else cv.get_feature_names()
    feature_arr = np.array(feature_names)
    
    n_clusters, n_words = top_words.shape
    cluster_words = [feature_arr[top_words[i, :]] for i in range(n_clusters)]
    
    # Dynamically size column widths with 3 spaces of padding between columns
    col_widths = [max(len(f"Cluster {i}"), max(len(str(w)) for w in cluster_words[i])) + 3 for i in range(n_clusters)]
    
    headers = [f"Cluster {i}".ljust(col_widths[i]) for i in range(n_clusters)]
    table = [
        "".join(headers),
        "-" * sum(col_widths)
    ]
    
    for j in range(n_words):
        row_words = [str(cluster_words[i][j]).ljust(col_widths[i]) for i in range(n_clusters)]
        table.append("".join(row_words))
        
    return "\n".join(table)

    
def prune_dict(D: Dict[Any, Any], keys_to_prune: Iterable[Any]) -> Dict[Any, Any]:
    """
    Deletes the matching keys from a dictionary in-place.
    """
    for key in keys_to_prune:
        D.pop(key, None)
    return D


def get_word_freqs(X: Sequence[str], pruning: Union[int, float] = 0) -> Dict[str, int]:
    """
    Calculates the frequency of words in X. Can prune words that appear less
    than a specified threshold (fraction or absolute count).
    """
    freq: Dict[str, int] = {}
    for text in X:
        for token in text.split():
            freq[token] = freq.get(token, 0) + 1
    
    n = len(X)
    if isinstance(pruning, (int, float)) and pruning > 0:
        if pruning < 1:
            pruning = pruning * n
        keys = []
        for key, val in freq.items():
            if val < pruning:
                keys.append(key)
        prune_dict(freq, keys)
    
    return freq


if __name__ == '__main__':
    # These are just for testing
    from sklearn.feature_extraction.text import CountVectorizer
    
    # 1. Provide some sample text chunks (acting as our documents)
    test_documents = [
        "Machine learning is a subset of artificial intelligence.",
        "It allows computers to learn from data.",
        "Supervised Learning: Models are trained on labeled data.",
        "Unsupervised Learning: Models find patterns in unlabeled data.",
        "Reinforcement Learning: Models learn by receiving rewards or penalties."
    ]
    
    print("--- 1. Testing Raw Tokenizer Output ---")
    tokenizer = NLP_Tokenizer(min_length=2, stem=True)
    
    # NLP_Tokenizer's transform returns a list of strings (joined tokens)
    processed_docs = tokenizer.transform(test_documents)
    
    for i, doc in enumerate(processed_docs):
        print(f"Doc {i+1}: {doc}")
        
    print("\n--- 2. Testing Integration with CountVectorizer ---")
    # We can pass our NLP_Tokenizer into Scikit-Learn Pipelines or Vectorizers!
    # Note: CountVectorizer does its own tokenization by default, but we can override it 
    # to just split by space since our tokenizer already outputs space-separated tokens.
    cv = CountVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x)
    
    # Fit the vectorizer on our pre-tokenized documents
    count_matrix = cv.fit_transform(processed_docs)
    
    print(f"Vocabulary Words: {cv.get_feature_names_out()}")
    print(f"Count Matrix Shape: {count_matrix.shape}")
    
    print("\n--- 3. Testing top_cluster_features ---")
    # Let's mock some cluster labels (e.g. Doc 0/1 are cluster 0, Doc 2/3/4 are cluster 1)
    mock_labels = np.array([0, 0, 1, 1, 1])
    
    # Get top 3 words for each of our 2 clusters
    top_words = top_cluster_features(count_matrix.toarray(), mock_labels, k=3)
    
    # Print a nice table!
    print("\nTop Words by Cluster:")
    print(table_top_words(cv, top_words))
