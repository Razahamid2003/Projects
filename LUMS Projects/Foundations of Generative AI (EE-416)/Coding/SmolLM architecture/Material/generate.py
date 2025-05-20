import torch
import torch.nn.functional as F
from torch import nn
import math

def __generate(model, inputs, num_tokens, tokenizer, max_length=50):
    """
    A basic greedy approach for text generation.
    """
    cache = []
    
    for _ in range(num_tokens):
        # generate logits over the entire vocab
        output = model(**inputs)
        
        # pick out the most likely token
        output_id = torch.argmax(output['logits'][0, -1]).item()
        
        # add it to context
        cache.append(output_id)
        
        # if end of sentence token is generated or max tokens have been generated, then return
        if output_id == tokenizer.eos_token_id or len(cache) >= max_length:
            break

        # Append the newly generated token to the context window
        new_token = torch.tensor([output_id], device=inputs['input_ids'].device)
        inputs['input_ids'] = torch.cat([inputs['input_ids'][0], new_token]).unsqueeze(0)
        # Update the attention mask (i.e., we can now let the model look at this newly generated token too)
        inputs['attention_mask'] = F.pad(inputs['attention_mask'], (0, 1), value=1)

    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(cache))


