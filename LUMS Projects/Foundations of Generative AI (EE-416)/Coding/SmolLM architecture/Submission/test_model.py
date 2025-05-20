import time 
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import smolConfig
from model import smolLM
from generate import __generate


# Load tokenizer and reference model
checkpoint = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
reference_model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Initialize smolLM
config = smolConfig()
test_model = smolLM(config)

# Load weights
state_dict = reference_model.state_dict()
missing_keys, unexpected_keys = test_model.load_state_dict(state_dict, strict=False)

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

start = time.perf_counter()
out = __generate(test_model, inputs, num_tokens=45, tokenizer=tokenizer)
end = time.perf_counter()

print('=='*10 + f' Output generated ({int(end-start)}) ' + '=='*10)
print(prompt + ' ' + out)

expected_output = "The future of AI is bright, but it’s not without its challenges. One of the biggest challenges is the lack of regulation and oversight. AI systems are often developed and deployed without the necessary safeguards in place to ensure they are safe and ethical."
result = prompt.lower().strip() + ' ' + out.lower().strip()

if result == expected_output.lower().strip():
    print(f'\nTest Passed! Output matches the expected output.')
else:
    print(f'\nTest Failed! Output does not match the expected output.')
