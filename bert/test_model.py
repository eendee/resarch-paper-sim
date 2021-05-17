import torch
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from functools import reduce
from nltk.tokenize import sent_tokenize

##############################################################################
#
# Load the stored model and run inferencing
#
##############################################################################

model_save_path = '../models/output/bert'


def count_average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)

def get_explanations(avg):
    threshold = 0.5
    if avg > threshold:
        print("The two paragraphs are similar contextually.")
    else:
        print("The two paragraphs are dissimilar contextually")



def get_similarity(para1, para2):
    model = SentenceTransformer(model_save_path)

    dic_para = {'paragraph1':para1, 'paragraph2':para2}
    source_paragraph = dic_para['paragraph1']
    target_paragraph = dic_para['paragraph2']
    source_sentences = sent_tokenize(source_paragraph)
    target_sentences = sent_tokenize(target_paragraph)

    # Compute embedding for both lists
    embeddings1 = model.encode(source_sentences, convert_to_tensor=True)
    embeddings2 = model.encode(target_sentences, convert_to_tensor=True)

    # Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    max_scores = []
    for scores in cosine_scores:
        max_elements, max_indices = torch.max(scores, dim=0)
        max_index = max_indices.item()
        max_scores.append(max_elements)

    avg = count_average(max_scores).item()
    return avg

if __name__ == '__main__':
    paragraph1 = 'Multi-objective optimization aims at finding trade-off solutions to conflicting objectives. These constitute the Pareto optimal set. In the context of expensive-to-evaluate functions, it is impossible and often non-informative to look for the entire set. As an end-user would typically prefer a certain part of the objective space, we modify the Bayesian multi-objective optimization algorithm which uses Gaussian Processes to maximize the Expected Hypervolume Improvement, to focus the search in the preferred region. The acumulated effects of the Gaussian Processes and the targeting strategy lead to a particularly efficient convergence to the desired part of the Pareto set. To take advantage of parallel computing, a multi-point extension of the targeting criterion is proposed and analyzed.'
    paragraph2 = 'We consider the problem of constrained multi-objective (MO) blackbox optimization using expensive function evaluations, where the goal is to approximate the true Pareto set of solutions satisfying a set of constraints while minimizing the number of function evaluations. We propose a novel framework named Uncertainty-aware Search framework for Multi-Objective Optimization with Constraints (USeMOC) to efficiently select the sequence of inputs for evaluation to solve this problem. The selection method of UseMOC consists of solving a cheap constrained MO optimization problem via surrogate models of the true functions to identify the most promising candidates and picking the best candidate based on a measure of uncertainty. We applied this framework to optimize the design of a multi-output switched-capacitor voltage regulator via expensive simulations. Our experimental results show that UseMOC is able to achieve more than 90 % reduction in the number of simulations needed to uncover optimized circuits.'
    score = get_similarity(paragraph1, paragraph2)
    get_explanations(score)


