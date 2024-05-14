from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
import jieba
import evaluate

import sys
sys.setrecursionlimit(100000)

distinct = evaluate.load("lsy641/distinct")

def compute_similar_metrics(eval_pred):
    predictions, labels = eval_pred

    # 字符级别
    # decoded_preds = [" ".join((pred.replace(" ", ""))) for pred in predictions]
    # decoded_labels = [" ".join((label.replace(" ", ""))) for label in labels]

    # 词级别
    decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in predictions]
    decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in labels]

    rouge = Rouge()

    bleu =np.array([0.,0.,0.,0.])
    weights = [(1.,0.,0.,0.),(1./2., 1./2.),(1./3., 1./3., 1./3.),(1./4., 1./4., 1./4., 1./4.)]
    for decoded_label, decoded_pred in zip(decoded_labels, decoded_preds):
        bleu +=np.array( sentence_bleu(
            references=[decoded_label.split(' ')],
            hypothesis=decoded_pred.split(' '),
            smoothing_function=SmoothingFunction().method1,weights=weights
        ))
    
    bleu /= len(decoded_labels)
    try:
        result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    except:
        print(f"RecursionError, response: {decoded_preds[0]}")
    result = {key: value['f'] * 100 for key, value in result.items()}
    result["bleu"] = {'bleu_1':bleu[0] * 100,'bleu_2':bleu[1] * 100,'bleu_3':bleu[2] * 100,'bleu_4':bleu[3] * 100}
    return result


def compute_diversity_metrics(pred):
    def calculate_distinct_n(tokens, n):
        n_grams = set()
        total_n_grams = 0
        for i in range(len(tokens) - n + 1):
            n_gram = tuple(tokens[i:i + n])
            n_grams.add(n_gram)
            total_n_grams += 1
        # print(n_grams)
        distinct_n = len(n_grams) / total_n_grams if total_n_grams > 0 else 0
        return distinct_n
    
    EAD = []
    distinct_1 = []
    distinct_2 = []
    distinct_3 = []
    distinct_4 = []
    for i, seq in enumerate(pred):
        token_list = list(jieba.cut(seq.replace(" ", "")))
        tokens = " ".join(token_list)
        result_dict = distinct.compute(predictions=[tokens], vocab_size=50000)
        EAD.append(result_dict["Expectation-Adjusted-Distinct"])
        distinct_1.append(result_dict["Distinct-1"])
        distinct_2.append(result_dict["Distinct-2"])
        distinct_3.append(result_dict["Distinct-3"])
        distinct_4.append(calculate_distinct_n(token_list, 4))
    
    mean_EAD = np.mean(EAD)
    mean_distinct_1 = np.mean(distinct_1)
    mean_distinct_2 = np.mean(distinct_2)
    mean_distinct_3 = np.mean(distinct_3)
    mean_distinct_4 = np.mean(distinct_4)
    
    return {
        "EAD": mean_EAD,
        "distinct_1": mean_distinct_1,
        "distinct_2": mean_distinct_2,
        "distinct_3": mean_distinct_3,
        "distinct_4": mean_distinct_4
    }



if __name__ == '__main__':
    result = compute_similar_metrics((['持续的学习是智慧的源泉，而实践则是知识的试金石。'], ['不懈的求知能够滋养心灵的土壤，而实际操作能够检验理论的真伪。']))
    print(result)
