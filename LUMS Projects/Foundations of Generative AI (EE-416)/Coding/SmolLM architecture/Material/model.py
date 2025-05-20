import torch.nn as nn
from layers import LlamaDecoder, RMSNorm

class smolModel(nn.Module):
    def __init__(self, config):
        # ===================== DO NOT CHANGE THE INIT FUNCTION! =====================
        super().__init__()
        # embedding layer which maps each token to a vector embedding
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size
        )

        # Stack of decoder layers (LlamaDecoder) defined by the configuration
        self.layers = nn.ModuleList([
            LlamaDecoder(config) for _ in range(config.num_hidden_layers)
        ])

        # RMSNorm: final layer normalization applied to hidden states
        self.norm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, input_ids=None, attention_mask=None):
        # ===================== DO NOT CHANGE THE FUNCTION ARGUMENTS! =====================
        # WRITE YOUR CODE HERE
        
        pass
    
class smolLM(nn.Module):
    # ===================== DO NOT CHANGE THE INIT FUNCTION! =====================
    """
    This is the Language Model. 
    It passes the embeddings from the SmolLM backbone into a LM head.
    The LM head generates logits over the space of the entire vocabulary for next word prediction.
    """
    def __init__(self, config):
        super().__init__()
        # SmolLM backbone which generates the contextualised embeddings for the input tokens
        self.model = smolModel(config)
        # The LM head which maps embeddings to logits over the vocabulary
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # weights between LM head and the token_embedding layer are shared in the SmolLM architecture
        self.tie_weights()

    def tie_weights(self):
        # ===================== DO NOT CHANGE THE FUNCTION ARGUMENTS! =====================
        # WRITE YOUR CODE HERE

        pass

    def forward(self, input_ids, attention_mask):
        # ===================== DO NOT CHANGE THE FUNCTION ARGUMENTS! =====================
        # WRITE YOUR CODE HERE

        logits = None
        
        return {'logits': logits}