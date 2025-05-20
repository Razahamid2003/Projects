from dataclasses import dataclass

# ===================== DO NOT CHANGE THE CONFIG FILE! =====================
@dataclass 
class smolConfig:
    vocab_size = 49152
    hidden_size = 576
    intermediate_size = 1536
    num_hidden_layers = 30
    num_heads = 9
    kv_heads = 3