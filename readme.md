# Para Executar

```
python experiments/run_experiment.py
```

1 - Crie um ambiente virtual </br>
2 - Instale as dependencias </br>
### Se tiver usando GPU

pip install torch --index-url https://download.pytorch.org/whl/cu118


## Arquivo Config.py

```python
# config.py

CHATGPT_MODEL = "gpt-4o-mini"
CHATGPT_API_KEY = "SUA_CHAVE_OPENAI"

SABIA_MODEL = "sabia-model"
SABIA_API_KEY = "SUA_CHAVE_SABIA"
SABIA_ENDPOINT = "https://api.sabia.ai/v1/chat"

TEMPERATURE = 0
MAX_RETRIES = 2


```