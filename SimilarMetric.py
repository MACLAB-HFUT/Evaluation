from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
import jieba

import sys
sys.setrecursionlimit(100000)

def compute_metrics(eval_pred):
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


if __name__ == '__main__':
    result = compute_metrics((['持续的学习是智慧的源泉，而实践则是知识的试金石。'], ['不懈的求知能够滋养心灵的土壤，而实际操作能够检验理论的真伪。']))
    print(result)
