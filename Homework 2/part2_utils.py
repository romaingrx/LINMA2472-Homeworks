import os
import pandas as pd
from functools import partial
from typing import Sequence, List
from spacy.tokens.doc import Doc
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec

from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags  # strip html tags
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation, strip_non_alphanum
from nltk.stem import WordNetLemmatizer


class AttributeDict(dict):
    """Like dict but with attribute access and setting"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def get_df_from_dir(dataset_dir: str) -> dir:
    DEFAULT_COLUMNS = ["sentence", "class", "class_idx"]
    dataset = AttributeDict(
        train=pd.DataFrame(columns=DEFAULT_COLUMNS),
        test=pd.DataFrame(columns=DEFAULT_COLUMNS)
    )

    classes = {
        "Donald Trump": "Donald-Trump-%s.csv",
        "Joe Biden": "Joe-Biden-%s.csv"
    }

    for typ in dataset.keys():  # train, test
        for class_idx, (class_name, class_path) in enumerate(classes.items()):  # Donald Trump, Joe Biden
            df = pd.read_csv(os.path.join(dataset_dir, class_path % typ), index_col=0)
            df["class"] = class_name  # Get the name (Donald Trump or Joe Biden)
            df["class_idx"] = class_idx  # 0 -> Donald Trump, 1 -> Joe Biden
            df.columns = DEFAULT_COLUMNS  # Force columns name
            dataset[typ] = dataset[typ].append(df, ignore_index=True)

    return dataset


# Tokenizer


lemmatize = WordNetLemmatizer().lemmatize


def tokenize(sentence: str, spacy_model, to_merge_entities: Sequence[str] = ["GPE", "LOC", "PERSON"]) -> Doc:
    doc = spacy_model(sentence)

    # Retrieve entities to merge
    ent_to_split = {ent.text: ent.text.split(' ') for ent in doc.ents if ent.label_ in to_merge_entities}

    # Set all entities lemma to the merged name
    for complete_entity, splitted_entity in ent_to_split.items():
        merged_entity = "_".join(splitted_entity)
        sentence.replace(complete_entity, merged_entity)

    CUSTOM_FILTERS = [lambda x: x.lower(),
                      strip_non_alphanum,
                      strip_punctuation,
                      # remove_stopwords,
                      strip_multiple_whitespaces]
    parsed_line = preprocess_string(sentence, CUSTOM_FILTERS)
    parsed_line = [lemmatize(x) for x in parsed_line]

    return parsed_line


def filter_tokens(tokens: Sequence[str], to_avoid: Sequence[str]) -> List[str]:
    returned_tokens = set()
    for token in tokens:
        if len(token) > 1 and (token not in to_avoid):
            returned_tokens.add(token)
    return list(returned_tokens)


def sent_preprocess(sentence: str, spacy_model, to_avoid: Sequence[str]) -> List[str]:
    pipe = (partial(tokenize, spacy_model=spacy_model),
            partial(filter_tokens, to_avoid=to_avoid))
    x = sentence
    for f in pipe:
        x = f(x)
    return x


def df_preprocess(df: pd.DataFrame, column: str, inplace: bool = False,
                  spacy_model_name: str = "en_core_web_sm") -> pd.DataFrame:
    import spacy
    import string
    from functools import partial

    try:
        nlp = spacy.load(spacy_model_name)
    except OSError as os_error:
        import sys
        import warnings
        import subprocess
        warnings.warn(f"spacy model {spacy_model_name} was not yet installed. Install it now.", ResourceWarning)
        subprocess.check_call([sys.executable, "-m", "spacy", "download", spacy_model_name])
        df_preprocess(df=df, column=column, inplace=inplace, spacy_model_name=spacy_model_name)

    to_avoid = list(string.punctuation + ' ') + ['\n', '\t']

    try:  # Speedup preprocess with parallelization
        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=False, verbose=0)
        preprocessed_column = df[column].parallel_apply(partial(sent_preprocess, spacy_model=nlp, to_avoid=to_avoid))
    except AttributeError:  # not parallelized
        preprocessed_column = df[column].apply(partial(sent_preprocess, spacy_model=nlp, to_avoid=to_avoid))

    preprocessed_df = df if inplace else df.copy(deep=True)

    preprocessed_df["preprocessed"] = preprocessed_column

    return preprocessed_df


word_set = {
    "Donald Trump": set(),
    "Joe Biden": set()
}


def __make_docs(df_row):
    global word_set
    tokens = df_row.preprocessed
    for token in tokens:
        word_set[df_row["class"]].add(token)
    doc = TaggedDocument(words=tokens, tags=[df_row["class"]])
    return doc


def df_make_doc(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    global word_set
    for key in word_set:
        word_set[key] = set()

    preprocessed_column = df.apply(__make_docs, axis=1)

    preprocessed_df = df if inplace else df.copy(deep=True)

    preprocessed_df["doc"] = preprocessed_column

    return preprocessed_df