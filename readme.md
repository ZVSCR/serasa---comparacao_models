# Para Executar

```
python -m experiments.run_experiment
```

1 - Crie um ambiente virtual </br>
2 - Instale as dependencias </br>
### Se tiver usando GPU

pip install torch --index-url https://download.pytorch.org/whl/cu118


## Arquivo Config.py

```python
# config.py

CHATGPT_MODEL = "gpt-4.1-mini"
CHATGPT_API_KEY = "SUA_CHAVE_OPENAI"

SABIA_MODEL = "sabia-model"
SABIA_API_KEY = "SUA_CHAVE_SABIA"
SABIA_ENDPOINT = "https://api.sabia.ai/v1/chat"

USE_MOCK = True


```