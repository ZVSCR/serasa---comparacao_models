# debate/debate_engine.py

import json
from debate.prompts import build_debate_prompt
from classifiers.chatgpt import classify_chatgpt
from classifiers.sabia import classify_sabia

def debate_round_chatgpt(text, self_result, other_result):

    from classifiers.chatgpt import client
    from config import CHATGPT_MODEL, TEMPERATURE

    prompt = build_debate_prompt(
        text,
        json.dumps(self_result, indent=2),
        json.dumps(other_result, indent=2)
    )

    response = client.chat.completions.create(
        model=CHATGPT_MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    return parsed

def llm_debate(text):

    gpt_initial = classify_chatgpt(text)
    sabia_initial = classify_sabia(text)

    gpt_revision = debate_round_chatgpt(
        text,
        gpt_initial,
        sabia_initial
    )

    # aqui pode implementar revisão SabIA igual

    # decisão final
    if gpt_revision["prediction"] == sabia_initial["prediction"]:
        final_prediction = gpt_revision["prediction"]
        strategy = "consensus_after_debate"
    else:
        if gpt_revision["confidence"] > sabia_initial["confidence"]:
            final_prediction = gpt_revision["prediction"]
        else:
            final_prediction = sabia_initial["prediction"]

        strategy = "confidence_tiebreak"

    return {
        "initial_round": {
            "chatgpt": gpt_initial,
            "sabia": sabia_initial
        },
        "debate_revision": {
            "chatgpt": gpt_revision
        },
        "final": {
            "prediction": final_prediction,
            "strategy": strategy
        }
    }
