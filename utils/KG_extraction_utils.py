import spacy
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from collections import Counter

# --- CONFIGURATION & PARAMETERS ---
PARAMS = {
    "spacy_model": "en_core_web_sm",
    "embedding_model": "all-MiniLM-L6-v2", # Best speed/accuracy for 2026
    "min_cluster_size": 2,                 # Minimum mentions to form an entity
    "sim_threshold": 0.85,                 # Semantic similarity for merging
    "lexical_weight": 0.3,                 # Weight of string overlap in distance
    "extraction_level": "rich",            # Options: 'basic' (only action verbs), 'rich' (+ properties, state-of-being)
}

# --- THE EXTRACTOR (Linguistic Rules) ---
def get_full_phrase(token, lemmatize=False):
    """
    Expands a token to include its modifiers (like adjectives and compounds).
    Also includes prepositional phrases for more context.
    Example: 'topology' -> 'topology of the space'
    """
    # Base modifiers: adjectives, compounds, and numeric modifiers
    phrase_tokens = [w for w in token.subtree if w.dep_ in ("amod", "compound", "flat", "nummod")]
    
    # Optionally include prepositional phrases attached to this noun
    # e.g., "topology of the space"
    for child in token.rights:
        if child.dep_ == "prep":
            phrase_tokens.extend([w for w in child.subtree])
            
    phrase_tokens.append(token)
    
    # Sort by index to maintain original order
    sorted_tokens = sorted(list(set(phrase_tokens)), key=lambda x: x.i)
    
    if lemmatize:
        return " ".join([w.lemma_.lower() for w in sorted_tokens if not w.is_punct])
    return " ".join([w.text for w in sorted_tokens if not w.is_punct])

def extract_triples(doc, extract_full_phrases=False, level="rich"):
    """
    Uses Dependency Parsing to find (Subject, Predicate, Object).
    Level 'basic': captures only standard action verbs (Subject -> Verb -> Object).
    Level 'rich': adds copular relationships (is-a), state-of-being, and attributes.
    """
    triples = []
    
    for token in doc:
        # Case 1: Action verbs (Subject -> Verb -> Object)
        # Standard across most KG extraction pipelines
        if token.pos_ == "VERB":
            # ...existing code...
            # (No changes to Case 1 logic, just wrapping)
            subj = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            obj = [w for w in token.rights if w.dep_ in ("dobj", "pobj", "attr", "oprd")]
            
            # Handle Passive Voice: "The space is filled by eigenvectors"
            if token.dep_ == "pass" or any(w.dep_ == "nsubjpass" for w in token.lefts):
                # We can try to find the agent (the "by ..." part)
                agent = [w for w in token.rights if w.dep_ == "agent"]
                if agent:
                    pobjs = [w for w in agent[0].rights if w.dep_ == "pobj"]
                    if subj and pobjs:
                        # Swap roles: eigenvectors (agent) -> fill -> space (subjpass)
                        obj = subj
                        subj = pobjs

            if subj and obj:
                for s in subj:
                    for o in obj:
                        subject_text = get_full_phrase(s) if extract_full_phrases else s.text.strip()
                        subject_lemma = get_full_phrase(s, lemmatize=True) if extract_full_phrases else s.lemma_.lower()
                        object_text = get_full_phrase(o) if extract_full_phrases else o.text.strip()
                        object_lemma = get_full_phrase(o, lemmatize=True) if extract_full_phrases else o.lemma_.lower()
                        
                        triples.append({
                            "subject": subject_text,
                            "subject_lemma": subject_lemma,
                            "relation": token.lemma_.lower(),
                            "object": object_text,
                            "object_lemma": object_lemma,
                            "sentence": token.sent.text
                        })

        # Case 2: Copular verbs / State of Being (Subject -> is -> Attribute)
        # Enabled only in 'rich' extraction level
        if level == "rich" and token.lemma_ == "be" and token.pos_ == "AUX":
            subj = [w for w in token.lefts if w.dep_ == "nsubj"]
            # ...existing code...
            attr = [w for w in token.rights if w.dep_ in ("attr", "acomp")]
            
            if subj and attr:
                for s in subj:
                    for a in attr:
                        subject_text = get_full_phrase(s) if extract_full_phrases else s.text.strip()
                        subject_lemma = get_full_phrase(s, lemmatize=True) if extract_full_phrases else s.lemma_.lower()
                        object_text = get_full_phrase(a) if extract_full_phrases else a.text.strip()
                        object_lemma = get_full_phrase(a, lemmatize=True) if extract_full_phrases else a.lemma_.lower()
                        
                        triples.append({
                            "subject": subject_text,
                            "subject_lemma": subject_lemma,
                            "relation": "is",
                            "object": object_text,
                            "object_lemma": object_lemma,
                            "sentence": token.sent.text
                        })
                        
    return triples

# --- THE RESOLVER (Vector Normalization) ---
def resolve_entities(mentions, model, min_size=2, similarity_threshold=0.85):
    """
    Clusters raw text mentions into 'Canonical Entities' using embeddings.
    For a whole book, this is the 'Global Normalization' step.
    """
    if not mentions: return {}
    
    # 1. Generate Embeddings for all unique mentions in the book
    unique_mentions = list(set(mentions))
    embeddings = model.encode(unique_mentions)
    
    # 2. Density-based Clustering (HDBSCAN)
    # This finds natural clusters (like "linear operator" and "linear transformations")
    clusterer = HDBSCAN(min_cluster_size=min_size, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings)
    
    entity_map = {}
    for i, label in enumerate(cluster_labels):
        mention = unique_mentions[i]
        if label == -1: 
            # Noise in HDBSCAN: Check for very high cosine similarity to existing clusters 
            # or treat as unique. For now, keep as unique.
            entity_map[mention] = mention 
        else:
            # Name the cluster by the shortest/most representative mention
            cluster_mentions = [unique_mentions[j] for j, l in enumerate(cluster_labels) if l == label]
            canonical_name = min(cluster_mentions, key=len) 
            entity_map[mention] = canonical_name 
            
    return entity_map

# --- THE GRAPH BUILDER ---
def build_graph(triples, entity_map):
    """
    Creates a NetworkX graph from triples, mapping subjects/objects to their
    canonical entities. When merging book-wide, this combines all findings.
    """
    G = nx.MultiDiGraph()
    
    for t in triples:
        # Use lemmatized form to look up the 'canonical' entity
        u = entity_map.get(t['subject_lemma'], t['subject_lemma'])
        v = entity_map.get(t['object_lemma'], t['object_lemma'])
        
        # Further normalization: if the lemma maps to a cluster canonical name
        u = entity_map.get(u, u)
        v = entity_map.get(v, v)
        
        rel = t['relation']
        
        # Metadata is preserved to allow for global analysis (e.g., edge weights)
        G.add_edge(u, v, label=rel, source_text=t['sentence'])
        
    return G

# --- THE DOCUMENT PRE-PROCESSOR (Chunking & PDF Handling) ---
def chunk_text(text, max_sentences=5, nlp=None):
    """
    Splits long text into manageable 'Context-Aware' chunks.
    For Knowledge Graphs, we prefer sentence-aligned chunks to avoid
    cutting a relationship in half.
    
    If you have a whole book, you can pass the entire string here, 
    but it's often safer to process Chapter by Chapter.
    """
    # Reuse an existing NLP object if provided to avoid multiple loads
    if nlp is None:
        nlp = spacy.load(PARAMS["spacy_model"], disable=["ner", "lemmatizer", "textcat"])
    
    # For exceptionally large strings (>1M chars), spaCy might need its max_length adjusted
    nlp.max_length = max(nlp.max_length, len(text) + 100)
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks = []
    
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i : i + max_sentences])
        chunks.append(chunk)
        
    return chunks

# --- MAIN EXECUTION FLOW ---
def run_knowledge_pipeline(text_corpus, extract_full_phrases=False, level=None, embedder=None, nlp=None):
    """
    Processes a list of strings (each a full document or sentence).
    For full documents, we use spaCy's sentence splitter.
    
    Parameters:
        text_corpus: List of strings or chunks
        extract_full_phrases: If True, uses chunking/modifiers for descriptive nodes
        level: 'basic' or 'rich' extraction
        embedder: (Optional) Pre-initialized SentenceTransformer model
        nlp: (Optional) Pre-initialized spaCy NLP model
    """
    extraction_level = level or PARAMS.get("extraction_level", "rich")
    print(f"Initializing Models (Level: {extraction_level})...")
    
    # Use existing models if passed, otherwise initialize defaults
    if nlp is None:
        nlp = spacy.load(PARAMS["spacy_model"])
        
    if embedder is None:
        embedder = SentenceTransformer(PARAMS["embedding_model"])
    
    # Step 1: Extract raw triples
    print("Extracting Triples...")
    all_raw_triples = []
    for text in text_corpus:
        # Process document once to handle sentence splitting
        doc = nlp(text)
        for sent in doc.sents:
            # Pass the sentence 'Span' directly to reuse pre-parsed data
            all_raw_triples.extend(extract_triples(sent, extract_full_phrases, extraction_level))
    
    # Step 2: Resolve Entities (Nodes)
    print("Resolving Entities...")
    # Use lemmatized forms for better clustering/keying
    all_mentions = [t['subject_lemma'] for t in all_raw_triples] + [t['object_lemma'] for t in all_raw_triples]
    entity_resolver = resolve_entities(all_mentions, embedder, PARAMS["min_cluster_size"])
    
    # Step 3: Assembly
    print("Assembling Graph...")
    kg = build_graph(all_raw_triples, entity_resolver)
    
    return kg

# Example Usage
if __name__ == "__main__":
    # corpus = [
    #     "The linear operator transforms the vector space.",
    #     "Bounded operators preserve the topology of the space.",
    #     "A linear transformation maps vectors to eigenvalues."
    # ]
    corpus = [
        """
        The linear operator transforms the vector space.
        Bounded operators preserve the topology of the space.
        A linear transformation maps vectors to eigenvalues.
        The vector space is infinite-dimensional.
        Eigenvectors fill the space.
        """,
        """
        The topology of the space is complex.
        Linear transformations can be bounded or unbounded.
        Bounded operators are important in functional analysis.
        Unbounded operators often arise in quantum mechanics.
        """
    ]
    
    # Setting extract_full_phrases=True to capture qualifiers like 'linear' and 'vector'
    graph = run_knowledge_pipeline(corpus, extract_full_phrases=True, level="rich")
    
    print(f"\nFinal Graph: {graph.number_of_nodes()} Nodes, {graph.number_of_edges()} Edges")
    for u, v, data in graph.edges(data=True):
        print(f"({u}) ---[{data['label']}]---> ({v})")