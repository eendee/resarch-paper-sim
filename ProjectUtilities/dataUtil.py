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
        self._df_paragraphs = None


    def getMeta(self):
        if self._df_meta is None:
            with open('../dataset/corpus_meta.pkl', 'rb') as f:
                self._df_meta = pickle.load(f)
        return self._df_meta


    def getSentences(self):
        if self._df_sentences is None:
            with open('../dataset/dataframe_of_sentences_2000', 'rb') as f:
                self._df_sentences = pickle.load(f)
        return self._df_sentences


    def get_paragraphs(self):
        if self._df_paragraphs is None:
            with open('../dataset/df_of_paragraphs_v50', 'rb') as f:
                self._df_paragraphs = pickle.load(f)
        return self._df_paragraphs


reader = DataReader()
decimal.getcontext().prec = 4


def get_similarity(vector1, vector2):
    cos_sim = np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))
    return cos_sim


def load_dataset_meta():
    return reader.getMeta()


def load_dataset():
    return reader.getSentences()


def load_dataset_paragraph():
    return reader.get_paragraphs()


def get_paper_df_by_id(doi, paragraph_id=None, use_paragraph_df = False):
    paper_id = doi.split('v')[0]
    if use_paragraph_df:
        df_all = load_dataset_paragraph()
        df_paper = df_all[df_all['paperId'] == paper_id].sort_values(['paragraph']).copy()
    else:
        df_all = load_dataset()
        df_paper = df_all[df_all['paperId'] == paper_id].sort_values(['paragraph', 'sentence_id']).copy()

    # print(df_paper.head())
    if paragraph_id is not None:
        # print(paragraph_id)
        df_paper = df_paper[df_paper['paragraph'] == int(paragraph_id)].copy()
    return df_paper


def get_paper_by_id(doi):
    df_paper = get_paper_df_by_id(doi)
    return df_paper.groupby('paragraph')['sentence'].apply(lambda tags: ' '.join(tags)).to_frame()
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


def get_compare_with_tm(source, target):

    source_paper_df = get_paper_df_by_id(source)
    target_paper_df = get_paper_df_by_id(target)

    unique_source_paragraphs = source_paper_df.paragraph.unique()
    # unique_target_paragraphs = target_paper_df.paragraph.unique()
    results = []
    for source_paragraph_id in unique_source_paragraphs:

        _r = get_similarity_values_for_paragraph(source, target_paper_df, source_paragraph_id, True)
        results.append(_r)

    return results


def get_common_words(source_paragraph_list, target_paragraph_list):
    return  list(set(source_paragraph_list) & set(target_paragraph_list))


def get_compare_by_paragraph(source, target, source_paragraph_id, target_paragraph_id=None):
    target_paper_df = get_paper_df_by_id(target, target_paragraph_id)
    return get_similarity_values_for_paragraph(source, target_paper_df, source_paragraph_id)


def get_similarity_values_for_paragraph(source, target_paper_df, paragraph_id, fast_mode = False):
    source_paragraph_df = get_paper_df_by_id(source, paragraph_id)

    if source_paragraph_df['paragraph'].count() == 0:
        return {'empty': True}

    source_paragraph = source_paragraph_df.groupby('paragraph')['sentence'].apply(lambda tags: ' '.join(tags)).values[0]

    _r = []

    unique_target_paragraphs = target_paper_df.paragraph.unique()
    for target_paragraph_id in unique_target_paragraphs:

        target_paragraph_df = target_paper_df[target_paper_df['paragraph'] == target_paragraph_id]
        target_paragraph = target_paragraph_df.groupby('paragraph')['sentence'].apply(lambda tags: ' '.join(tags)).values[0]

        if fast_mode:
            _r.append({
                'score': '1',
            })

        else:
            topic_model_sim = tpm.sim(source_paragraph, target_paragraph)
            topic_model_explanation = tpm.explanations(source_paragraph, target_paragraph)

            doc2vec_vals = compute_doc2vec_paragrpah_sim(source_paragraph_df, target_paragraph_df)
            doc2vec_exp = get_doc2vec_explanation(doc2vec_vals)

            doc2vec_p, common_words = compute_doc2vec_paragraph_level(source, target_paper_df.paperId.values[0], paragraph_id, target_paragraph_id)

            # co_occurrence_score = sf.jaccard_similarity(source_paragraph, target_paragraph)



            # bert_score = bert.get_similarity(source_paragraph, target_paragraph)

            ttr = sf.ttr(source_paragraph, target_paragraph)

            _r.append({
                'doc2vecParagraph': {
                    'score': decimal.Decimal(str(doc2vec_p)),
                    'explanation': ''
                },
                'doc2vec':{
                    'score':decimal.Decimal(str(doc2vec_vals[0])),
                    'containment': doc2vec_vals[1],
                    'explanation': doc2vec_exp
                },
                'topicModel': {
                    'score':topic_model_sim,
                    'explanation': topic_model_explanation
                },
                'commonWords': common_words,
                'ttr': ttr

            })
    return _r


def compute_doc2vec_paragraph_level(source, target,  source_paragraph_id,  target_paragraph_id):
    source_paragraph_vector = get_paper_df_by_id(source, source_paragraph_id, True)
    target_paragraph_vector = get_paper_df_by_id(target, target_paragraph_id, True)

    sim_val = 0
    common_words = []
    if source_paragraph_vector.count != 0 and target_paragraph_vector.count != 0:
        sim_val = get_similarity(target_paragraph_vector.paragraph_vector.values[0],
                              source_paragraph_vector.paragraph_vector.values[0])
        common_words = get_common_words(source_paragraph_vector.tags.values[0], target_paragraph_vector.tags.values[0])

    return sim_val, common_words


def compute_doc2vec_paragrpah_sim(source_paragraph_df, target_paragraph_df):
    source_unique_sentences = source_paragraph_df.sentence_id.unique()
    max_values = []
    for source_sentence_id in source_unique_sentences:
        # print(source_paragraph_df.head())
        source_sentence_df = source_paragraph_df[source_paragraph_df['sentence_id'] == source_sentence_id]
        source_vector = source_sentence_df.iloc[0].doc2vec
        # col = 'sim' + str(target_paragraph_id) + '_' + str(source_sentence_id)
        target_paragraph_df['sim'] = target_paragraph_df.doc2vec.apply(get_similarity, args=[source_vector])
        max_value_for_sentence = target_paragraph_df.sort_values(by='sim', ascending=False).iloc[0]['sim']
        # avg_value_for_sentence = sum(target_paragraph_df.sim.values)/len(target_paragraph_df.sim.values)
        max_values.append(max_value_for_sentence)
    sentence_containment = [1 if x > 0.4 else 0 for x in max_values]
    avg_sim_score_for_paragraph = reduce(lambda x, y: x + y, max_values) / len(max_values)

    return avg_sim_score_for_paragraph, sentence_containment


def get_doc2vec_explanation(doc2vec_vals):

    score = round(doc2vec_vals[0], 2)
    number_entailed = len([x for x in doc2vec_vals[1] if x > 0.4])
    total_count = len(doc2vec_vals[1])

    exp1 = 'The source and target paragraph vectors ' \
           'have an average similarity of ' + str(score * 100) + "%"

    exp2 = ''
    if number_entailed == total_count:
        exp2 = 'The source paragraph is completely contained in the target'
    elif number_entailed == 0:
        exp2 = 'The source paragraph does not seem to semantically contain any sentence in the target paragraphs'
    else:
        exp2 = 'Target paragraph semantically contains ' + str(number_entailed) + ' out of '\
               + str(total_count) + ' sentences in the source paragraph'

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
    source_paragraph = source_paragraph_df.groupby('paragraph')['sentence'].apply(lambda tags: ' '.join(tags)).values
    print(source_paragraph)

