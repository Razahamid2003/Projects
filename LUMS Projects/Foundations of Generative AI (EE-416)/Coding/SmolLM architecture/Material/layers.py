import torch 
import torch.nn as nn 
from attention import GroupedQueryAttention


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Root Mean Square (RMS) Normalization Layer.

        RMSNorm normalizes the input tensor by its root mean square (RMS) value 
        instead of the variance, as done in LayerNorm. It is particularly useful 
        in transformer models for stabilizing training while avoiding mean-based 
        normalization.

        Args:
            hidden_size (int): The number of features in the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. 
                                Defaults to 1e-6.

        Attributes:
            weight (torch.Parameter): Learnable scaling parameter of shape (hidden_size,).
        """
        # ===================== DO NOT CHANGE THE INIT FUNCTION! =====================
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # Learnable scaling factor
        self.variance_epsilon = eps

    def forward(self, x):
        # ===================== DO NOT CHANGE THE FUNCTION ARGUMENTS! =====================
        # WRITE YOUR CODE HERE
        
        pass

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        """
        Gated Multi-Layer Perceptron (MLP) from the LLaMA Architecture.

        This MLP block is used in LLaMA models and incorporates a gating mechanism with 
        the SiLU (Sigmoid Linear Unit) activation function. The gating mechanism helps 
        improve expressiveness while maintaining computational efficiency.

        Args:
            hidden_size (int): The input and output feature size of the MLP.
            intermediate_size (int): The size of the hidden layer, typically larger than `hidden_size`.

        Attributes:
            gate (nn.Linear): Linear layer for the gating mechanism.
            up_proj (nn.Linear): Linear layer for the transformation.
            down_proj (nn.Linear): Linear layer to project back to `hidden_size`.
            activation (nn.SiLU): SiLU activation function.
        """
        # ===================== DO NOT CHANGE THE INIT FUNCTION! =====================
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = nn.modules.activation.SiLU()

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

    def forward(self, x):
        # ===================== DO NOT CHANGE THE FUNCTION ARGUMENTS! =====================
        # WRITE YOUR CODE HERE

        pass
        
    
class LlamaDecoder(nn.Module):
    def __init__(self, config):
        # ===================== DO NOT CHANGE THE INIT FUNCTION! =====================
        """
        This is the Llama decoder block.
        """
        super().__init__()
        # Self Attention Module
        self.self_attn = GroupedQueryAttention(config)

        # FFN Module
        self.mlp = MLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size)

        # Pre Attention and Post Attention normalisation
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-05)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, x, attention_mask):
        # ===================== DO NOT CHANGE THE FUNCTION ARGUMENTS! =====================
        # WRITE YOUR CODE HERE
        
        pass