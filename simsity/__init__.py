import mmh3 
from diskcache import Cache
from queue import LifoQueue
import datetime as dt
import itertools as it
from pathlib import Path
from typing import Iterable, Protocol, Callable, Union, Any, Dict, Optional

import srsly
from hnswlib import Index
from tqdm import tqdm

DB_NAME = "db.gz.json"
INDEX_NAME = "index.bin"
METADATA_NAME = "metadata.json"


def create_hash(thing):
    return mmh3.hash(str(thing), signed=False)


class Transformer(Protocol):
    def transform(self, X, y=None):
        pass


EncType = Union[Callable, Transformer]


class SimSityIndex:
    """Object for easy querying."""

    def __init__(self, index: Index, encoder: EncType, cache: Cache) -> None:
        self.index = index
        self.encoder = encoder
        self.cache = cache

    def query(self, query: Union[str, Dict], n: int = 10):
        """
        Query using approximate nearest neighbors

        The object handles the encoder/data from disk.
        """
        arr = encode_data(self.encoder, query)
        return self.query_vector(query=arr, n=n)

    def query_vector(self, query: Union[str, Dict], n: int = 10):
        """Query using a vector."""
        labels, distances = self.index.knn_query(query, k=n)
        out = [self.db[label] for label in labels[0]]
        return out, list(distances[0])

    def walk(self, *args, n=10, depth=3, uniq_id=lambda d: d):
        """Walk through the index, finding nearest neighbors of nearest neighbors.

        Arguments:

        - args: the queries to start the walk off with
        - n : number of items to return per query
        - depth: how deep should the search go
        - uniq_id: function that can determine the uniqness of the item (must be hashable)
        """
        q = LifoQueue()
        seen = {}

        for i in range(depth):
            new_args = []

            for arg in args:
                res, dists = self.index.query(arg, n=n)
                for item in res:
                    q.put(item)

            if depth != 0:
                while not q.empty():
                    item = q.get()
                    if uniq_id(item) not in seen:
                        yield item
                        new_args.append(item)
                        seen[uniq_id(item)] = 1
            args = new_args
    
    def to_disk(self, path: Union[str, Path]):
        """Store the index to disk, with metadata"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        index.save_index(str(path / INDEX_NAME))
        metadata = {
            "created": str(dt.datetime.now())[:19],
            "dim": dim,
            "n_items": len(data),
            "space": space,
            "encoder": str(encoder),
        }
        srsly.write_json(
            path / METADATA_NAME,
            metadata,
        )


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


def encode_data(encoder, data):
    if callable(encoder):
        return encoder(data)
    else:
        return encoder.transform([data])


def create_index(
    data: list,
    encoder: Union[Transformer, Callable],
    path: Union[Path, str],
    space: str = "cosine",
    pbar: bool = True,
    batch_size: int = 500,
):
    """
    Creates a simple ANN index. Uses hnswlib under the hood.
    You need to provide a scikit-learn compatible encoder.
    """
    index = None
    dim = 0
    batches = batch(data, batch_size)
    cache = Cache(path)
    
    if pbar:
        batches, batches_copy = it.tee(batches)
        total = sum(1 for _ in batches_copy)
        batches = tqdm(batches, desc="indexing", total=total)
    for batch in batches:
        hash_table = {
            create_hash(item): item 
            for item in batch 
            if create_hash(item) not in cache
        }
        encoded = encode_data(encoder, list(hash_table.values()))
        if not index:
            # We need to know the size of the vectors before we can init the database.
            dim = encoded.shape[1]
            index = Index(space=space, dim=dim)
            index.init_index(max_elements=len(data))
        index.add_items(encoded, ids=hash_table.key())
        with Cache(cache.directory) as reference:
            for key, value in hash_table.items():
                reference.set(key, value)
    if not index:
        raise RuntimeError(
            "Something has gone terrible wrong. There is no index. Did you supply data?"
        )
    index = SimSityIndex(index=index, encoder=encoder, cache=cache)
    index.to_disk(path)
    return index


def load_index(path: Union[str, Path], encoder: EncType):
    """Load in a simsity index from a path. Must supply same encoder."""
    path = Path(path)
    metadata = srsly.read_json(path / METADATA_NAME)
    index = Index(space=metadata["space"], dim=metadata["dim"])
    index.load_index(str(path / INDEX_NAME))
    db = {int(i): k for i, k in srsly.read_gzip_json(path / DB_NAME).items()}
    return SimSityIndex(index=index, encoder=encoder, db=db)
