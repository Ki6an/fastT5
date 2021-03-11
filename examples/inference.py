from fastT5 import OnnxT5, export_and_get_onnx_model, get_onnx_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastT5.model_testing_tools import speed_test

model_or_model_path = 't5-small'

model = export_and_get_onnx_model(model_or_model_path)

# if you've already exported the models 
# model = get_onnx_model(model_or_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_or_model_path)

t_input = "translate English to French: The universe is a dark forest."

token = tokenizer(t_input, return_tensors='pt')

input_ids = token['input_ids']
attention_mask = token['attention_mask']

# 'set num_beams = 1' for greedy search
tokens = model.generate(input_ids=input_ids,
                        attention_mask=attention_mask, num_beams=2)

output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)

print(output)


# # for speed testing...
# pt_model = AutoModelForSeq2SeqLM.from_pretrained(model_or_model_path)
# speed_test(model, pt_model, range(1, 6, 1), range(10, 200, 10))
