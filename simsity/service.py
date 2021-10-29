import json
import pathlib

import pandas as pd
from joblib import dump, load
from simsity import __version__
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class Service:
    """
    This object represents a nearest neighbor lookup service. You can
    pass it an encoder and a method to index the data.

    Arguments:
        encoder: A scikit-learn compatible encoder for the input.
        indexer: A compatible indexer for the nearest neighbor search.
        storage: A dictionary containing the data to be retreived with index. Meant to be ignored by humans.
    """

    def __init__(self, encoder, indexer, storage=None) -> None:
        self.encoder = encoder
        self.indexer = indexer
        self.storage = storage if storage else {}

        self._trained = self._check_is_trained()
        self._indexed = False

    def _check_is_trained(self):
        """Check if the encoder has all components fit"""
        if isinstance(self.encoder, BaseEstimator):
            try:
                check_is_fitted(self.encoder)
                return True
            except NotFittedError:
                return False
        elif isinstance(self.encoder, Pipeline):
            is_fitted = []
            for step in self.encoder:
                try:
                    check_is_fitted(step)
                    is_fitted.append(True)
                except NotFittedError:
                    is_fitted.append(False)
            
            return all(is_fitted)

    def train_from_dataf(self, df, features=None):
        """
        Trains the service from a dataframe.

        Arguments:
            df: Pandas DataFrame that contains text to train the service with.
            features: Names of the features to encode.
        """
        subset = df
        if features:
            subset = df[features]
        self.storage = {i: r for i, r in enumerate(subset.to_dict(orient="records"))}
        try:
            data = self.encoder.transform(subset)
        except NotFittedError:
            data = self.encoder.fit_transform(subset)

        self.indexer.index(data)
        self._trained = True
        self._indexed = True

        return self

    def train_from_transformed_data(self, df, transformed_data, features=None):
        """
        Index the data.
        
        Arguments:
            df: Pandas DataFrame that contains text to show when doing NN queries.
            features: Names of the features to show.
            transformed_data: Data which has already been transformed by the encoder.
        """
        subset = df
        if features:
            subset = df[features]
        self.storage = {i: r for i, r in enumerate(subset.to_dict(orient="records"))}

        self.indexer.index(transformed_data)
        self._indexed = True

    def query(self, n_neighbors=10, out="list", data=None, **kwargs):
        """
        Query the service.

        Arguments:
            n_neighbors: Number of neighbors to return.
            out: Output format. Can be either "list" or "dataframe".
            data: Data which has already been transformed by the encoder.
            kwargs: Arguments to pass as the query.
        """
        if not self._trained:
            raise RuntimeError("Cannot query, Service is not trained.")

        if n_neighbors > len(self.storage):
            raise ValueError(
                "n_neighbors cannot be greater than the number of items in the storage."
            )
        
        if data is None:
            try:
                data = self.encoder.transform(pd.DataFrame([{**kwargs}]))
            except NotFittedError:
                RuntimeError("Cannot query, Service is not trained.")

        if self._indexed:
            idx, dist = self.indexer.query(data, n_neighbors=n_neighbors)
        else:
            raise RuntimeError("Cannot query, Service is not indexed.")
        
        res = [
            {"item": self.storage[idx[i]], "dist": float(dist[i])}
            for i in range(len(idx))
        ]
        if out == "list":
            return res
        if out == "dataframe":
            return pd.DataFrame([{**r["item"], "dist": r["dist"]} for r in res])

    def save(self, path):
        """
        Save the service

        Arguments:
            path: Path to the folder to save the service to.
        """
        if not self._trained:
            raise RuntimeError("Cannot save, Service is not trained.")
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        storage_path = pathlib.Path(path) / "storage.json"
        storage_path.write_text(json.dumps(self.storage))
        metadata_path = pathlib.Path(path) / "metadata.json"
        metadata_path.write_text(json.dumps({"version": __version__}))
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
        metadata_path = pathlib.Path(path) / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata["version"] != __version__:
            raise RuntimeError(
                f"Version mismatch. Expected {__version__}, got {metadata['version']}"
            )
        storage_path = pathlib.Path(path) / "storage.json"
        storage = {int(k): v for k, v in json.loads(storage_path.read_text()).items()}
        encoder = load(pathlib.Path(path) / "encoder.joblib")
        decoder = load(pathlib.Path(path) / "indexer.joblib")
        service = cls(encoder, decoder, storage)
        service._trained = True
        return service

    def serve(self, host, port=8080):
        """
        Start a server for the service.

        Once the server is started, you can `POST` the service using the following URL:

        ```
        http://<host>:<port>/query
        ```

        Arguments:
            host: Host to bind the server to.
            port: Port to bind the server to.
        """
        import uvicorn
        from simsity.serve import create_app

        uvicorn.run(create_app(self), host=host, port=port)
