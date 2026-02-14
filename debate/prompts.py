# debate/prompts.py

SYSTEM_CLASSIFY = """
Você é especialista em detecção de fake news.
Responda apenas em JSON válido:

{
  "prediction": 0 ou 1,
  "label_name": "fake" ou "real",
  "confidence": número entre 0 e 1,
  "justification": "explicação objetiva"
}
"""

def build_debate_prompt(text, self_result, other_result):

    return f"""
Texto:
{text}

Sua classificação anterior:
{self_result}

Classificação do outro modelo:
{other_result}

Reavalie sua decisão considerando os argumentos apresentados.

Responda apenas em JSON válido no mesmo formato.
"""
