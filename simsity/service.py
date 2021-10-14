import json
import pathlib

import pandas as pd
from joblib import dump, load


class Service:
    """
    Super Simple Similarities Service

    Arguments:
        encoder: A scikit-learn compatible encoder for the input.
        indexer: A compatible indexer for the nearest neighbor search.
        storage: A dictionary containing the data to be retreived with index. Meant to be ignored by humans.
    """

    def __init__(self, encoder, indexer, storage={}) -> None:
        self.encoder = encoder
        self.indexer = indexer
        self.storage = storage
        self.__trained = False

    def train_text_from_dataf(self, df, text_col="text"):
        """
        Trains the service from a dataframe assuming text as input.

        Arguments:
            df: Pandas DataFrame that contains text to train the service with.
            text_col: Name of the column containing text.
        """
        texts = list(df[text_col])
        self.storage = {i: {"text": t} for i, t in enumerate(texts)}
        data = self.encoder.fit_transform(texts)
        self.indexer.index(data)
        self.__trained = True
        return self

    def train_from_dataf(self, df, features=None):
        """
        Trains the service from a dataframe.

        Arguments:
            df: Pandas DataFrame that contains text to train the service with.
            features: Name of the column containing text.
        """
        subset = df
        if features:
            subset = df[features]
        self.storage = {i: r for i, r in enumerate(subset.to_dict(orient="records"))}
        data = self.encoder.fit_transform(subset)
        self.indexer.index(data)
        self.__trained = True
        return self

    def query_text(self, text, n_neighbors=10):
        """
        Query the service
        """
        data = self.encoder.transform([text])
        idx, dist = self.indexer.query(data, n_neighbors=n_neighbors)
        return [
            {"item": self.storage[idx[0][i]], "dist": dist[0][i]}
            for i in range(idx.shape[1])
        ]

    def query(self, n_neighbors=10, **kwargs):
        """
        Query the service
        """
        print(pd.DataFrame([{**kwargs}]))
        data = self.encoder.transform(pd.DataFrame([{**kwargs}]))
        print(data)
        idx, dist = self.indexer.query(data, n_neighbors=n_neighbors)
        return [
            {"item": self.storage[idx[0][i]], "dist": dist[0][i]}
            for i in range(idx.shape[1])
        ]

    def save(self, path):
        """
        Save the service

        Arguments:
            path: Path to the folder to save the service to.
        """
        if not self.__trained:
            raise Exception("Cannot save, Service is not trained.")
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        storage_path = pathlib.Path(path) / "storage.json"
        storage_path.write_text(json.dumps(self.storage))
        dump(self.encoder, pathlib.Path(path) / "encoder.joblib")
        dump(self.indexer, pathlib.Path(path) / "indexer.joblib")

    @classmethod
    def load(cls, path):
        """
        Loads a service

        Arguments:
            path: Path to the folder to load the service from.
        """
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"{path} does not exist")
        storage_path = pathlib.Path(path) / "storage.json"
        storage = {int(k): v for k, v in json.loads(storage_path.read_text()).items()}
        encoder = load(pathlib.Path(path) / "encoder.joblib")
        decoder = load(pathlib.Path(path) / "indexer.joblib")
        return cls(encoder, decoder, storage)
