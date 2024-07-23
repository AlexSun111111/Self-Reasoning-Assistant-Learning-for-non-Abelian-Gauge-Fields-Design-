from transformers import T5Tokenizer, T5Model

# Constants
DEFAULT_T5_NAME = 't5-small'

# Load the T5 model and tokenizer
t5_model_cache = {}
t5_tokenizer_cache = {}

def get_t5_model(name=DEFAULT_T5_NAME):
    if name not in t5_model_cache:
        t5_model_cache[name] = T5Model.from_pretrained(name)
    return t5_model_cache[name]

def get_t5_tokenizer(name=DEFAULT_T5_NAME):
    if name not in t5_tokenizer_cache:
        t5_tokenizer_cache[name] = T5Tokenizer.from_pretrained(name)
    return t5_tokenizer_cache[name]

# Function to encode text using T5
def t5_encode_text(texts, name=DEFAULT_T5_NAME):
    tokenizer = get_t5_tokenizer(name)
    model = get_t5_model(name)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state

# Function to get the encoded dimension of the T5 model
def get_encoded_dim(name=DEFAULT_T5_NAME):
    model = get_t5_model(name)
    return model.config.d_model
