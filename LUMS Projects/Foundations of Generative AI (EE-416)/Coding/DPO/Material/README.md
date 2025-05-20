# DPO: Direct Preference Optimization of SmolLM

## Motivation: The Preference Alignment Problem

Aligning large language models (LLMs) with human preferences is traditionally done using Reinforcement Learning with Human Feedback (RLHF). However, RLHF has serious challenges:

- **Reward Model Complexity**: Training a reward model introduces extra steps and potential inaccuracies
- **Training Instability**: Reinforcement learning (e.g., PPO) is unstable, sensitive to hyperparameters
- **High Cost**: RL-based methods require expensive computation
- **Implementation Overhead**: RL algorithms are harder to debug and slower to converge

## Enter DPO: Simple Supervised Fine-tuning from Preferences

**Direct Preference Optimization (DPO)** offers a simpler solution:  
Instead of using reinforcement learning, DPO directly fine-tunes the language model based on human preferences, using **only supervised learning**.

### Core Concept: Prefer Chosen Responses Over Rejected Ones

Given a dataset of:

- a **prompt** `x`
- a **chosen** response `y⁺`
- a **rejected** response `y⁻`

DPO trains the model `π` to **assign a higher probability** to `y⁺` than `y⁻` under the prompt `x`.

No rewards. No RL. Just comparing log-probabilities and optimizing.

---

## DPO Loss

The DPO objective is:

```math
\mathcal{L}_{\text{DPO}} = -\log\left( \sigma\left( \beta \left[ (\log \pi_\theta(y_c|x) - \log \pi_\theta(y_r|x)) - (\log \pi_{\text{ref}}(y_c|x) - \log \pi_{\text{ref}}(y_r|x)) \right] \right) \right)
```

where:

- $x$: the prompt
- $y_c$: the chosen response
- $y_r$: the rejected response
- $\pi_\theta$: the current policy model
- $\pi_{\text{ref}}$: the frozen reference model
- $\beta$: a temperature-like hyperparameter (typically 0.1–0.5)
- $\sigma$: the sigmoid function

---

## DPO in Practice: Applying to SmolLM

For this assignment:

1. **Load two models**:
   - **Policy Model**: the model you will fine-tune (initialized from fine-tuned weights)
   - **Reference Model**: a frozen copy of the original fine-tuned model
2. **Freeze the reference model** to prevent updates
3. **Freeze parts of the policy model** to speed up training on limited hardware
4. **Load the preference dataset**: (prompt, chosen response, rejected response)
5. **Compute the DPO loss** by implementing the given functions
6. **Train the policy model** to minimize the DPO loss

The reference model helps stabilize training, but DPO mainly relies on the contrast between chosen and rejected responses.

---

## Advantages of DPO for SmolLM

- **No Reward Model Needed**: Simpler setup compared to RLHF
- **No RL Instability**: Stable supervised learning
- **Easy to Implement**: Standard forward-backward pass
- **Efficient**: Lower compute and faster convergence
- **Strong Performance**: Can match RLHF results on preference tasks

---

## Hyperparameters

Important hyperparameters:

- **Beta (β)**:
  - Controls sharpness of preference.
  - Higher `β` → stronger push toward preferred outputs
- **Batch Size**:
  - You may adjust to fit your hardware
- **Learning Rate**:
  - Typically lower than for full fine-tuning
- **Frozen Layers**:
  - To speed up training if needed

---

## Performance Goals

At the end of training:

- **Preference accuracy should improve**:  
  Your fine-tuned policy model should prefer human-preferred responses more often than the base model.

You will measure:

- Preference accuracy before DPO
- Preference accuracy after DPO

---

## In This Assignment

You will:

1. Load and prepare SmolLM models
2. Process and clean the preference dataset
3. Implement DPO loss
4. Fine-tune SmolLM using preference data
5. Evaluate the improvement in preference accuracy

---

## Additional Resources

### Original Paper

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023

### Helpful Video

- [Direct Preference Optimization (DPO) explained: Bradley-Terry model, log probabilities, math](https://youtu.be/hvGa5Mba4c8?si=TD9y5qOM_7PZXtqK) - Umar Jamil

---
