from fastT5 import (OnnxT5, export_and_get_onnx_model,
                    get_onnx_model, get_onnx_runtime_sessions,
                    generate_onnx_representation, quantize)

from transformers import AutoTokenizer

model_or_model_path = 't5-small'

model = export_and_get_onnx_model(model_or_model_path)
tokenizer = AutoTokenizer.from_pretrained(model_or_model_path)


def test_translation():
    t_input = "translate English to French: The universe is a dark forest."
    token = tokenizer(t_input, return_tensors='pt')

    input_ids = token['input_ids']
    attention_mask = token['attention_mask']

    tokens = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask, num_beams=2)

    output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)

    assert output == "L'univers est une forêt sombre."


# add some more tests
