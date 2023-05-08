from embetter.text import SentenceEncoder
from embetter.grab import KeyGrabber

from simsity import create_index, load_index
from simsity.datasets import fetch_recipes
from sklearn.pipeline import make_pipeline


# Fetch data
df_recipes = fetch_recipes()
recipes = df_recipes["text"][:1000]

# Create an encoder
encoder = SentenceEncoder()


def check_output(texts):
    for text in texts:
        assert "pork" in text


def test_base_usage(tmpdir):
    # Make an index with a path
    index = create_index(recipes, encoder, path=tmpdir)
    out1, _ = index.query("pork")
    check_output(out1)

    # Load an index from a path
    loader_index = load_index(path=tmpdir, encoder=encoder)
    out2, _ = loader_index.query("pork")
    check_output(out2)
    assert out1 == out2


def test_callable_usage(tmpdir):
    # You can also pass a callable as an encoder
    index = create_index(recipes, lambda d: encoder.transform(d), path=tmpdir)
    out1, _ = index.query("pork")
    check_output(out1)

    # This as well
    loader_index = load_index(path=tmpdir, encoder=encoder)
    out2, _ = loader_index.query("pork")
    check_output(out2)
    assert out1 == out2


def test_dict_data_usage(tmpdir):
    recipe_stream = [{"text": t} for t in recipes]
    print(recipe_stream[0])
    pipe = make_pipeline(KeyGrabber("text"), encoder)

    index = create_index(data=recipe_stream, encoder=pipe, path=tmpdir)
    out1, _ = index.query({"text": "pork"})
    check_output([ex['text'] for ex in out1])

    # This as well
    loader_index = load_index(path=tmpdir, encoder=pipe)
    out2, _ = loader_index.query({"text": "pork"})
    check_output([ex['text'] for ex in out2])
    assert out1 == out2