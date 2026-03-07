from nltk.translate.bleu_score import sentence_bleu
from typing import List

class MachineTranslationMetric:
    def __init__(self) -> None:
        pass

    def bleu_score(self, predictions: List[str], labels: List[str]) -> float:
        total_score = 0.0
        for prediction, label in zip(predictions, labels):
            total_score += sentence_bleu([label.split(" ")], prediction.split(' '))
        return total_score / len(predictions)