import re
import gensim
import pickle
from gensim.parsing.preprocessing import remove_stopwords
import nltk


class Doc2Vec:

    def __init__(self, path_to_model):
        self.model = None
        self.load_model(path_to_model)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def get_model(self):
        return self.model

    def get_vector(self, sentence):
        tokens = get_tokens(sentence)
        return self.model.infer_vector(tokens)


def get_tokens(sentence):
    sentence = remove_stopwords(sentence)
    tokens = gensim.utils.simple_preprocess(sentence)
    return tokens


def pre_process(paragraph):
    paragraph = re.sub(r'\S*@\S*\s?', '', paragraph, flags=re.MULTILINE)  # remove email
    paragraph = re.sub(r'http\S+', '', paragraph, flags=re.MULTILINE)  # remove web addresses
    paragraph = paragraph.replace("et al.", "")
    return paragraph


def get_sentences_from_paragraph(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return sentences
