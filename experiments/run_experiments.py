from orchestrator.evaluator import evaluate_news
from utils.logger import log_result

def run(dataset):

    for i, text in enumerate(dataset):

        result = evaluate_news(text, i)

        log_result(result)

        print(f"Notícia {i} processada.")
