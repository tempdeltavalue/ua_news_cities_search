import numpy as np
import time
import random
from typing import List, Dict, Set, Tuple, Optional

# Try importing CuPy. If not available (no GPU), code will need handling.
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Warning: CuPy not found. GPU functions will fail.")

# --- 1. TEXT PROCESSING & VECTORIZATION FUNCTIONS ---

def extract_ngrams(s: str, n: int = 3) -> List[str]:
    """Extracts N-grams from a string with padding."""
    s = " " + s.lower() + " "
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def build_vocabulary(strings: List[str], n: int = 3) -> Tuple[Dict[str, int], List[str]]:
    """
    Builds a unique vocabulary of N-grams from a list of strings.
    Returns: (ngram_to_index_dict, sorted_ngram_list)
    """
    all_ngrams: Set[str] = set()
    for s in strings:
        all_ngrams.update(extract_ngrams(s, n))

    ngram_list = sorted(list(all_ngrams))
    ngram_to_index = {ngram: i for i, ngram in enumerate(ngram_list)}
    return ngram_to_index, ngram_list

def strings_to_matrix(strings: List[str], ngram_map: Dict[str, int], n: int = 3) -> np.ndarray:
    """
    Converts a list of strings into a CPU Numpy frequency matrix.
    Shape: (num_strings, vocab_size)
    """
    M = len(strings)
    D = len(ngram_map)
    # Using int32 for frequency counts
    matrix = np.zeros((M, D), dtype=np.int32)

    for i, s in enumerate(strings):
        ngrams = extract_ngrams(s, n)
        for ngram in ngrams:
            if ngram in ngram_map:
                matrix[i, ngram_map[ngram]] += 1
    return matrix

# --- 2. GPU CALCULATION ENGINE ---

def compute_cosine_similarity_gpu(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """
    Computes Cosine Distance matrix on GPU using CuPy.
    
    Args:
        matrix_a: Query matrix (M x D)
        matrix_b: Reference/Database matrix (N x D)
        
    Returns:
        distances_cpu: Numpy array of shape (M, N) where values are (1 - similarity).
    """
    if not HAS_GPU:
        raise RuntimeError("CuPy is required for GPU calculations.")
    
    # 1. Transfer data to GPU
    A_gpu = cp.asarray(matrix_a)
    B_gpu = cp.asarray(matrix_b)
    
    # 2. Compute Numerator (Dot Product) -> Shape (M, N)
    dot_product_gpu = A_gpu @ B_gpu.T
    
    # 3. Compute Denominator (Norms)
    # Norm A shape: (M, 1)
    norm_A_gpu = cp.sqrt(cp.sum(A_gpu**2, axis=1, keepdims=True))
    # Norm B shape: (N, ) -> broadcasted later
    norm_B_gpu = cp.sqrt(cp.sum(B_gpu**2, axis=1))
    
    norm_product_gpu = norm_A_gpu * norm_B_gpu
    
    # 4. Compute Similarity & Distance
    similarity_gpu = cp.zeros_like(dot_product_gpu, dtype=float)
    
    # Avoid division by zero
    valid_mask = norm_product_gpu > 1e-6
    similarity_gpu[valid_mask] = dot_product_gpu[valid_mask] / norm_product_gpu[valid_mask]
    
    # Distance = 1 - Similarity
    distances_gpu = 1.0 - similarity_gpu
    
    # 5. Return to CPU
    cp.cuda.Stream.null.synchronize() # Ensure calculations are done
    return cp.asnumpy(distances_gpu)

# --- 3. DATA GENERATION UTILS (For Testing) ---

UKR_ROOTS = [
    "Київ", "Львів", "Одеса", "Харків", "Дніпро", "Запоріжжя", "Вінниця", "Суми",
    "Рівне", "Миколаїв", "Херсон", "Бахчисарай", "Нахімовський", "Ленінський"
]
NOISE_CHARS = "qwertyzxv"

def apply_noise(name: str, num_noise: int) -> str:
    """Inserts random characters into a string."""
    name_list = list(name)
    for _ in range(num_noise):
        idx = random.randint(0, len(name_list))
        name_list.insert(idx, random.choice(NOISE_CHARS))
    return "".join(name_list)

def generate_dummy_data(num_queries: int = 1000, num_db: int = 30000) -> Tuple[List[str], List[str], List[str], Dict[str, str]]:
    """
    Generates synthetic Ukrainian city names and noisy queries.
    Returns: (queries_list, database_list, noisy_targets, truth_map)
    """
    np.random.seed(42)
    random.seed(42)
    
    search_bases = ["Одеса", "Бахчисарайський", "Львова", "Херсон", "Київський Район"]
    
    # 1. Generate Database (Cities)
    db_list = []
    db_list.extend(search_bases)
    while len(db_list) < num_db:
        root = random.choice(UKR_ROOTS)
        suffix = random.choice(['ська', 'ський', 'е', 'ий', ''])
        db_list.append(root + suffix)
    
    random.shuffle(db_list)
    db_list = db_list[:num_db]
    
    # 2. Generate Queries (Random + Noisy Targets)
    queries = []
    # Fill with random stuff first
    while len(queries) < num_queries - len(search_bases):
        root = random.choice(UKR_ROOTS)
        suffix = random.choice(['ська', 'инський', 'е'])
        queries.append(root + suffix)
        
    # Add specific noisy words we want to track
    truth_map = {}
    noisy_targets = []
    
    for name in search_bases:
        noisy_name = apply_noise(name, random.randint(1, 3))
        noisy_targets.append(noisy_name)
        truth_map[noisy_name] = name
        
    queries.extend(noisy_targets)
    random.shuffle(queries)
    
    return queries, db_list, noisy_targets, truth_map