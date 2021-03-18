from fastT5 import (
    OnnxT5,
    export_and_get_onnx_model,
    get_onnx_model,
    get_onnx_runtime_sessions,
    generate_onnx_representation,
    quantize,
)

from transformers import AutoTokenizer

model_or_model_path = "t5-small"

# Step 1. convert huggingfaces t5 model to onnx
onnx_model_paths = generate_onnx_representation(model_or_model_path)

# Step 2. (recommended) quantize the converted model for fast inference and to reduce model size.
quant_model_paths = quantize(onnx_model_paths)

# step 3. setup onnx runtime
model_sessions = get_onnx_runtime_sessions(quant_model_paths)

# step 4. get the onnx model
model = OnnxT5(model_or_model_path, model_sessions)


#   --------common-part--------
tokenizer = AutoTokenizer.from_pretrained(model_or_model_path)

t_input = "translate English to French: The universe is a dark forest."

token = tokenizer(t_input, return_tensors="pt")

input_ids = token["input_ids"]
attention_mask = token["attention_mask"]
# 'set num_beams = 1' for greedy search
tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=2)

output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)

print(output)
