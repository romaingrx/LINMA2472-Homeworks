from characters_real_names import *

import re
import networkx as nx
import numpy as np

import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")

# Used to get the gender and distinct people with same family noun.
def get_person_title(span):
    if span.label_ == "PERSON" and span.start != 0:
        prev_token = span.doc[span.start - 1]
        if prev_token.text in ("Mr", "Mr.", "Mrs", "Mrs."):
            return prev_token.text + " "
    return ""

Span.set_extension("person_title", getter=get_person_title, force=True)

class HPBook():
    def __init__(self): 
        # ---
        # Get the book text data

        book = open("Homework 1\Harry Potter and the Sorcerer.txt").read()

        book_chapters = re.split(r'CHAPTER [\w+]+', book, flags=re.IGNORECASE)[1:]
        book_chapters[-1] = book_chapters[-1].split("THE END")[0]

        # ---
        # Cleaning and tokenization

        book_chapters_paragraphs = dict()
        real_charachers_by_paragraphs = list()

        for i, chapter in enumerate(book_chapters):

            # Split and clean up each paragraphs
            paragraphs = chapter.split('\n\n')

            for paragraph in paragraphs:
                clean_paragraph = re.sub(r"[\"\;\:\s]+", ' ', paragraph)

                # Tag and detect every person's name by paragraph
                tokens = nlp(clean_paragraph)
                paragraph_persons = [(ent._.person_title + ent.text)
                                    for ent in tokens.ents if ent.label_ == "PERSON"]

                # Correct all possible mentions of a person
                # And remove false positive

                real_paragraph_persons = []
                for person in paragraph_persons:
                    if person in persons_real_names:
                        real_paragraph_persons.append(person)
                    else:
                        for check, replace in name_to_replace.items():
                            reg = re.compile(check, )
                            if reg.match(person):
                                real_paragraph_persons.append(replace)
                                break

                real_charachers_by_paragraphs.append(set(real_paragraph_persons))

            book_chapters_paragraphs[i] = paragraphs

        # ---
        # Graph Filling
        # Add edge between 2 characters of the same paragraph
        graph = nx.Graph()

        for unique_paragraph_persons in real_charachers_by_paragraphs:
            if len(unique_paragraph_persons) < 2:
                continue
            graph.add_nodes_from(unique_paragraph_persons)
            unique = list(unique_paragraph_persons)
            for idx in range(len(unique)-1):
                edge = graph.get_edge_data(unique[idx], unique[idx-1])
                if edge is not None:
                    graph[unique[idx]][unique[idx-1]]['weight'] += 1
                else:
                    graph.add_edge(unique[idx], unique[idx-1], weight=1)

        
        self.graph = graph
        self.edges_weight = np.array([graph[u][v].get('weight', 1) for u, v in graph.edges])
        self.nodes_size = np.array([len(list(graph.neighbors(node))) for node in graph.nodes])
