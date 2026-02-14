# orchestrator/evaluator.py

from classifiers.bert import classify_bert
from debate.debate_engine import llm_debate

def evaluate_news(text, text_id):

    bert_result = classify_bert(text)
    llm_result = llm_debate(text)

    return {
        "text_id": text_id,
        "bert": bert_result,
        "llm_debate": llm_result,
        "agreement":
            bert_result["prediction"] ==
            llm_result["final"]["prediction"]
    }
