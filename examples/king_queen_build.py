import tqdm 
import spacy 
from simsity import create_index
from embetter.text import spaCyEncoder

# Use spaCy large english vectors for this one
nlp = spacy.load("en_core_web_lg")
encoder = spaCyEncoder(nlp)

# Collecting terms
pbar = tqdm.tqdm(nlp.vocab.vectors.keys(), desc="collecting terms")
terms = [nlp.vocab[i].text for i in pbar]

# Populate the ANN vector index and use it. 
index = create_index(terms, encoder, path="index_spacy")
