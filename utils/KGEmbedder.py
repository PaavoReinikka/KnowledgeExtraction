import torch
import spacy
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RichTriplet:
    subject: str
    relation: str
    obj: str
    sentence: str
    subj_offsets: Tuple[int, int]
    obj_offsets: Tuple[int, int]
    subj_lemma: str
    obj_lemma: str

class KnowledgeEmbedder:
    """Handles the high-precision token-level embeddings."""
    def __init__(self, model_name: str = "sentence-transformers/all-distilroberta-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def get_span_vector(self, last_hidden_states, token_offsets, char_start, char_end) -> torch.Tensor:
        relevant_indices = [
            i for i, (s, e) in enumerate(token_offsets) 
            if s is not None and e is not None and s >= char_start and e <= char_end
        ]
        if not relevant_indices: return last_hidden_states[0, 0, :]
        return torch.mean(last_hidden_states[0, relevant_indices, :], dim=0)

    def embed_batch(self, triplets: List[RichTriplet]) -> List[Dict]:
        results = []
        for t in triplets:
            inputs = self.tokenizer(t.sentence, return_tensors="pt", return_offsets_mapping=True)
            offsets = inputs.pop("offset_mapping")[0].tolist()
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            s_vec = self.get_span_vector(outputs.last_hidden_state, offsets, *t.subj_offsets)
            o_vec = self.get_span_vector(outputs.last_hidden_state, offsets, *t.obj_offsets)
            
            results.append({
                "triplet": t,
                "embeddings": {"subj": s_vec, "obj": o_vec}
            })
        return results

class KnowledgeGraphPipeline:
    def __init__(self, spacy_model="en_core_web_sm", use_transformer=True):
        self.nlp = spacy.load(spacy_model)
        self.use_transformer = use_transformer
        self.embedder = KnowledgeEmbedder() if use_transformer else None
        # Fallback/Lite embedder for HDBSCAN
        self.lite_embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _get_phrase_info(self, token, lemmatize=False) -> Tuple[str, str, Tuple[int, int]]:
        """Modified from your utility: returns text/lemma AND char offsets."""
        phrase_tokens = [w for w in token.subtree if w.dep_ in ("amod", "compound", "flat", "nummod")]
        for child in token.rights:
            if child.dep_ == "prep":
                phrase_tokens.extend([w for w in child.subtree])
        phrase_tokens.append(token)
        sorted_tokens = sorted(list(set(phrase_tokens)), key=lambda x: x.i)
        
        text = " ".join([w.text for w in sorted_tokens if not w.is_punct])
        lemma = " ".join([w.lemma_.lower() for w in sorted_tokens if not w.is_punct])
        
        # Calculate character offsets within the sentence
        start = sorted_tokens[0].idx - token.sent.start_char
        end = sorted_tokens[-1].idx + len(sorted_tokens[-1].text) - token.sent.start_char
        return text, lemma, (start, end)

    def extract_rich_triplets(self, text: str) -> List[RichTriplet]:
        doc = self.nlp(text)
        triplets = []
        for sent in doc.sents:
            for token in sent:
                # Case 1: Verbs & Case 2: Copular (is-a)
                is_verb = token.pos_ == "VERB"
                is_copula = token.lemma_ == "be" and token.pos_ == "AUX"
                
                if is_verb or is_copula:
                    subjs = [w for w in token.lefts if "subj" in w.dep_]
                    objs = [w for w in token.rights if w.dep_ in ("dobj", "pobj", "attr", "acomp")]
                    
                    for s in subjs:
                        for o in objs:
                            s_text, s_lemma, s_off = self._get_phrase_info(s)
                            o_text, o_lemma, o_off = self._get_phrase_info(o)
                            
                            triplets.append(RichTriplet(
                                subject=s_text, relation=token.lemma_.lower(), obj=o_text,
                                sentence=sent.text, subj_offsets=s_off, obj_offsets=o_off,
                                subj_lemma=s_lemma, obj_lemma=o_lemma
                            ))
        return triplets

    def resolve_entities_hdbscan(self, mentions: List[str]) -> Dict[str, str]:
        """Your original HDBSCAN strategy for global normalization."""
        if not mentions: return {}
        embeddings = self.lite_embedder.encode(list(set(mentions)))
        clusterer = HDBSCAN(min_cluster_size=2)
        labels = clusterer.fit_predict(embeddings)
        
        unique_m = list(set(mentions))
        entity_map = {}
        for i, label in enumerate(labels):
            if label == -1: entity_map[unique_m[i]] = unique_m[i]
            else:
                cluster = [unique_m[j] for j, l in enumerate(labels) if l == label]
                entity_map[unique_m[i]] = min(cluster, key=len)
        return entity_map
    
    def consolidate_nodes_by_vector(self, G: nx.MultiDiGraph, similarity_threshold: float = 0.92) -> nx.MultiDiGraph:
        """
        Scans all nodes in the graph. If two nodes have highly similar average vectors,
        merges them into the more frequent canonical name.
        """
        # 1. Map each node to its average vector (nodes might have multiple edges/contexts)
        node_vectors = {}
        for n in G.nodes():
            vectors = []
            # Check vectors from outgoing edges
            for _, _, d in G.out_edges(n, data=True):
                if 'subj_vec' in d: vectors.append(d['subj_vec'])
            # Check vectors from incoming edges
            for _, _, d in G.in_edges(n, data=True):
                if 'obj_vec' in d: vectors.append(d['obj_vec'])
                
            if vectors:
                node_vectors[n] = np.mean(vectors, axis=0)

        nodes = list(node_vectors.keys())
        if not nodes: return G

        # 2. Compute Cosine Similarity Matrix
        vec_matrix = np.array([node_vectors[n] for n in nodes])
        sim_matrix = cosine_similarity(vec_matrix)

        # 3. Find pairs to merge and store with their similarity scores
        # We use a list of tuples so we can sort by similarity
        merges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if sim_matrix[i, j] > similarity_threshold:
                    node_a, node_b = nodes[i], nodes[j]
                    # Heuristic: Target is the shorter/more canonical string
                    target, source = (node_a, node_b) if len(node_a) <= len(node_b) else (node_b, node_a)
                    merges.append((sim_matrix[i, j], source, target))

        # Sort by similarity descending so we merge the most certain pairs first
        merges.sort(key=lambda x: x[0], reverse=True)

        # 4. Perform the actual graph contraction
        processed_sources = set()
        for score, source, target in merges:
            # Check if source hasn't already been merged into something else
            if source in G and target in G and source not in processed_sources:
                print(f"Merging: '{source}' -> '{target}' (Similarity: {score:.4f})")
                
                # nx.contracted_nodes moves all edges from 'source' to 'target'
                # self_loops=False prevents 'Apple' -> 'Apple' edges
                G = nx.contracted_nodes(G, target, source, self_loops=False)
                processed_sources.add(source)
                
        return G

    def run(self, corpus: List[str], vector_refine: bool = True):
        # 1. Extraction: Get triplets and offsets
        all_triplets = []
        for chunk in corpus:
            all_triplets.extend(self.extract_rich_triplets(chunk))
        
        # 2. Embedding: High-precision span vectors
        # Results is a list of { 'triplet': T, 'embeddings': {'subj': vec, 'obj': vec} }
        embedded_data = self.embedder.embed_batch(all_triplets)
        
        # 3. Unified Entity Resolution
        # Instead of embedding strings, we group the high-quality vectors by lemma
        lemma_to_vectors = {}
        for item in embedded_data:
            t = item['triplet']
            for lemma, vec in [(t.subj_lemma, item['embeddings']['subj']), 
                            (t.obj_lemma, item['embeddings']['obj'])]:
                if lemma not in lemma_to_vectors:
                    lemma_to_vectors[lemma] = []
                lemma_to_vectors[lemma].append(vec.numpy())

        # Average the vectors for each unique lemma to get a "Global Lemma Vector"
        unique_lemmas = list(lemma_to_vectors.keys())
        avg_vectors = np.array([np.mean(lemma_to_vectors[l], axis=0) for l in unique_lemmas])
        
        # 4. HDBSCAN on the high-quality vectors
        clusterer = HDBSCAN(min_cluster_size=2, metric='euclidean')
        labels = clusterer.fit_predict(avg_vectors)
        
        entity_map = {}
        for i, label in enumerate(labels):
            if label == -1: 
                entity_map[unique_lemmas[i]] = unique_lemmas[i]
            else:
                cluster = [unique_lemmas[j] for j, l in enumerate(labels) if l == label]
                entity_map[unique_lemmas[i]] = min(cluster, key=len)

        # 5. Graph Construction
        G = nx.MultiDiGraph()
        for i, data in enumerate(embedded_data):
            t = data['triplet']
            u = entity_map.get(t.subj_lemma, t.subj_lemma)
            v = entity_map.get(t.obj_lemma, t.obj_lemma)
            
            G.add_edge(u, v, 
                       label=t.relation, 
                       subj_vec=data['embeddings']['subj'].numpy(),
                       obj_vec=data['embeddings']['obj'].numpy(),
                       context=t.sentence)
        
        # 6. Optional Vector-Based Consolidation
        if vector_refine:
            print("Performing vector-based consolidation...")
            G = self.consolidate_nodes_by_vector(G)

        return G

    def runV1(self, corpus: List[str], vector_refine: bool = True) -> nx.MultiDiGraph:
        # 1. Extraction
        all_triplets = []
        for chunk in corpus:
            all_triplets.extend(self.extract_rich_triplets(chunk))
        
        # 2. Embedding (The new robust way)
        embedded_data = self.embedder.embed_batch(all_triplets) if self.use_transformer else []
        
        # 3. Entity Resolution (Using your HDBSCAN logic on lemmas)
        all_mentions = [t.subj_lemma for t in all_triplets] + [t.obj_lemma for t in all_triplets]
        entity_map = self.resolve_entities_hdbscan(all_mentions)
        
        # 4. Graph Construction
        G = nx.MultiDiGraph()
        for i, data in enumerate(embedded_data):
            t = data['triplet']
            u = entity_map.get(t.subj_lemma, t.subj_lemma)
            v = entity_map.get(t.obj_lemma, t.obj_lemma)
            
            G.add_edge(u, v, 
                       label=t.relation, 
                       subj_vec=data['embeddings']['subj'].numpy(),
                       obj_vec=data['embeddings']['obj'].numpy(),
                       context=t.sentence)
        # 5. Optional Vector-Based Consolidation
        if vector_refine:
            print("Performing vector-based consolidation...")
            G = self.consolidate_nodes_by_vector(G)

        return G

# --- Execution ---
if __name__ == "__main__":
    pipeline = KnowledgeGraphPipeline()
    sample_text = ["Apple Inc. creates the iPhone. The tech giant is based in Cupertino.",
                   "The iPhone is a product of Apple. Cupertino is home to Apple Inc.",
                   "Apple's CEO is Tim Cook. Tim Cook leads Apple Inc."]
    graph = pipeline.run(sample_text)
    
    for u, v, d in graph.edges(data=True):
        print(f"Node '{u}' vector sample: {d['subj_vec'][:3]}...")
        print(f"Edge: ({u}) --[{d['label']}]--> ({v})")