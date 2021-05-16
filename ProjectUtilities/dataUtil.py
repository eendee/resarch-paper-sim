import os
import pickle
import numpy as np
from numpy.linalg import norm
from functools import reduce
import decimal
import models.TopicModel.tpm as tpm
import models.SyntacticFeatures.syntacticfeatures as sf


class DataReader:
    def __init__(self):
        self._df_sentences = None
        self._df_meta = None


    def getMeta(self):
        if self._df_meta is None:
            with open('../dataset/corpus_meta.pkl', 'rb') as f:
                self._df_meta = pickle.load(f)
        return self._df_meta


    def getSentences(self):
        if self._df_sentences is None:
            with open('../dataset/dataframe_of_sentences_2000.pkl', 'rb') as f:
                self._df_sentences = pickle.load(f)
        return self._df_sentences


reader = DataReader()
decimal.getcontext().prec = 4


def get_similarity(vector1, vector2):
    cos_sim = np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))
    return cos_sim


def load_dataset_meta():
    return reader.getMeta()


def load_dataset():
    return reader.getSentences()


def get_paper_df_by_id(doi, paragraph_id=None):
    paper_id = doi.split('v')[0]
    df_all = load_dataset()
    df_paper = df_all[df_all['paperId'] == paper_id].sort_values(['paragraph', 'sentence_id']).copy()

    print(df_paper.head())
    if paragraph_id is not None:
        # print(paragraph_id)
        df_paper = df_paper[df_paper['paragraph'] == int(paragraph_id)].copy()
    return df_paper


def get_paper_by_id(doi):
    df_paper = get_paper_df_by_id(doi)
    return df_paper.groupby('paragraph')['sentence'].apply(lambda tags: ','.join(tags)).to_frame()
    # paragraphs = list(df_paper.paragraph.unique)
    # for p in paragraphs:
    #     df_sents = df_paper[df_paper['paragraph'] == p]


def get_compare(source, target):

    source_paper_df = get_paper_df_by_id(source)
    target_paper_df = get_paper_df_by_id(target)

    unique_source_paragraphs = source_paper_df.paragraph.unique()
    # unique_target_paragraphs = target_paper_df.paragraph.unique()
    results = []
    for source_paragraph_id in unique_source_paragraphs:
        # results[str(source_paragraph_id)] = []
        _r = get_similarity_values_for_paragraph(source, target_paper_df, source_paragraph_id)
        results.append(_r)

    # for each paragraph in the target
    # for each sentence in the source paragraph
    # pick the max score after comparing each sentence in the current target paragraph
    return results


def get_compare_by_paragraph(source, target, source_paragraph_id, target_paragraph_id=None):
    target_paper_df = get_paper_df_by_id(target, target_paragraph_id)
    return get_similarity_values_for_paragraph(source, target_paper_df, source_paragraph_id)


def get_similarity_values_for_paragraph(source, target_paper_df, paragraph_id):
    source_paragraph_df = get_paper_df_by_id(source, paragraph_id)

    if source_paragraph_df['paragraph'].count() == 0:
        return {'empty': True}

    source_paragraph = source_paragraph_df.groupby('paragraph')['sentence'].apply(lambda tags: ','.join(tags)).values[0]

    _r = []

    unique_target_paragraphs = target_paper_df.paragraph.unique()
    for target_paragraph_id in unique_target_paragraphs:
        target_paragraph_df = target_paper_df[target_paper_df['paragraph'] == target_paragraph_id]
        target_paragraph = target_paragraph_df.groupby('paragraph')['sentence'].apply(lambda tags: ','.join(tags)).values[0]

        topic_model_sim = tpm.sim(source_paragraph, target_paragraph)
        topic_model_explanation = tpm.explanations(source_paragraph, target_paragraph)

        doc2vec_vals = compute_doc2vec_paragrpah_sim(source_paragraph_df, target_paragraph_df)
        doc2vec_exp = get_doc2vec_explanation(doc2vec_vals)

        co_occurrence_score = sf.jaccard_similarity(source_paragraph, target_paragraph)
        _r.append({
            'doc2vec_sim':{
                'score':decimal.Decimal(str(doc2vec_vals[0])),
                'matches': doc2vec_vals[1],
                'count': doc2vec_vals[2],
                'explanation': doc2vec_exp
            },
            'topic_model_sim': {
                'score':topic_model_sim,
                'explanation': topic_model_explanation
            },
            'syntactic_sim':{
                'score':co_occurrence_score
            }
        })
    return _r


def compute_doc2vec_paragrpah_sim(source_paragraph_df, target_paragraph_df):
    source_unique_sentences = source_paragraph_df.sentence_id.unique()
    max_values = []
    for source_sentence_id in source_unique_sentences:
        source_sentence_df = source_paragraph_df[source_paragraph_df['sentence_id'] == source_sentence_id]
        source_vector = source_sentence_df.iloc[0].vector
        # col = 'sim' + str(target_paragraph_id) + '_' + str(source_sentence_id)
        target_paragraph_df['sim'] = target_paragraph_df.vector.apply(get_similarity, args=[source_vector])
        max_value_for_sentence = target_paragraph_df.sort_values(by='sim', ascending=False).iloc[0]['sim']
        # avg_value_for_sentence = sum(target_paragraph_df.sim.values)/len(target_paragraph_df.sim.values)
        max_values.append(max_value_for_sentence)
    number_above_threshold = [x for x in max_values if x > 0.4]
    avg_sim_score_for_paragraph = reduce(lambda x, y: x + y, max_values) / len(max_values)

    return avg_sim_score_for_paragraph, len(number_above_threshold), len(max_values)


def get_doc2vec_explanation(doc2vec_vals):

    score = round(doc2vec_vals[0], 2)
    number_entailed = doc2vec_vals[1]
    total_count = doc2vec_vals[2]

    exp1 = 'The source and target paragraph vectors ' \
           'have an average similarity of ' + str(score * 100) + "%"

    exp2 = ''
    if number_entailed == total_count:
        exp2 = 'The source paragraph is completely entailed in the target'
    elif number_entailed == 0:
        exp2 = 'The source paragraph does not seem to semantically entail the target paragraphs'
    else:
        exp2 = 'Source paragraph semantically entails ' + str(number_entailed) + ' out of '\
               + str(total_count) + ' sentences in the target paragraph'

    return [exp1, exp2]



if __name__ == "__main__":
    df_test = load_dataset_meta()
    assert df_test.id.count() > 0, "No data in dataframe"

    test_paper = "1203.2394"
    #test_paper2 = "1610.05755"
    test_paper2 = "1203.2394"
    df_paper_test = get_paper_by_id(test_paper)
    assert df_paper_test.sentence.count() > 0, "No data in paper dataframe"

    #results = get_compare(test_paper, test_paper2)
    #print(len(results))

    source_paragraph_df = get_paper_df_by_id(test_paper2, "1")
    source_paragraph = source_paragraph_df.groupby('paragraph')['sentence'].apply(lambda tags: ','.join(tags)).values
    print(source_paragraph)

