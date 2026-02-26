from Classifier.bert import classify_bert

from config import USE_MOCK

if USE_MOCK:
    from Classifier.chatgpt_mock import classify_chatgpt
    from Classifier.sabia_mock import classify_sabia
else:
    from Classifier.chatgpt import classify_chatgpt
    from Classifier.sabia import classify_sabia


def resolve_llm(chatgpt_result, sabia_result):

    if chatgpt_result["prediction"] == sabia_result["prediction"]:
        return chatgpt_result["prediction"]

    return "conflict"

def evaluate_news(text, text_id):

    bert_result = classify_bert(text)
    gpt_result = classify_chatgpt(text)
    sabia_result = classify_sabia(text)

    llm_final = resolve_llm(gpt_result, sabia_result)

    return {
        "text_id": text_id,
        "bert": bert_result,
        "chatgpt": gpt_result,
        "sabia": sabia_result,
        "llm_final": llm_final,
        "agreement_with_bert":
            bert_result["prediction"] == llm_final
            if llm_final != "conflict"
            else False
    }