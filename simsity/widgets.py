import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML


def search_widget(**indices):
    """
    Loads up a jupyter widget to perform search. You'll need to make
    sure that ipywidgets are installed beforehand.
    
    Inputs:
        - indices: name:index kwargs
    
    Outputs:
        Jupyter Widget that allows you to "search" multiple indices
    """
    
    def reduce(q=""):
        data = {}
        for i, (name, index) in enumerate(indices.items()):
            texts, dists = index.query(q, n=10)
            data[f"{name}-text"] = texts
            data[f"dist-{i}"] = dists
        
        df = pd.DataFrame(data).loc[lambda d: ~d[f"dist-{i}"].isna()]
        if df.shape[0] != 0:
            display(HTML(df.to_html()))

    q = widgets.Text()

    out = widgets.interactive_output(reduce, {'q': q})

    return widgets.VBox([q, out])
