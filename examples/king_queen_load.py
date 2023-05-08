import tqdm 
import spacy 
from simsity import load_index
from embetter.text import spaCyEncoder

# Use spaCy large english vectors for this one
nlp = spacy.load("en_core_web_lg")
encoder = spaCyEncoder(nlp)

index = load_index("index_spacy", encoder)

def vec(term):
    return nlp(term).vector

q = vec("king") - vec("man") + vec("woman")
terms, dists = index.query_vector(q)
print(list(zip(terms, dists)))