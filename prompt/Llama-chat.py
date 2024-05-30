# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")

# 构建prompt
request = "What is the language model, and how does it work?"

prompt = f"""[INST] <<SYS>>
You are an expert machine learning engineer skillful in explaining complex concepts in a simple manner. In your explanations, you use examples and helpful analogies to make sure the audience understands the details. <</SYS>>
{request} [/INST]"""

inputs = tokenizer(request, return_tensors="pt").to('cuda')

outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=1024)

response = tokenizer.decode(outputs[0], skip_special_tokens=False)

print(response)

# #
# <s>[INST] <<SYS>>
# {your_system_message}
# <</SYS>>
# {user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2}[/INST]
# #


# sentence = "this is a example."
# input_ids = tokenizer.encode(sentence, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(input_ids=input_ids)
#     embedding = list(outputs.hidden_states)
