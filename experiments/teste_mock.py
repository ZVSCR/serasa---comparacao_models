from orchestrator.evaluator import evaluate_news

if __name__ == "__main__":

    texto_teste = """
    Governo anuncia nova medida econômica que reduz impostos.
    """

    resultado = evaluate_news(texto_teste, text_id=1)

    print("\nResultado completo:\n")
    print(resultado)