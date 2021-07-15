import os
import tempfile
import unittest

from transformers import AutoTokenizer

from fastT5 import (export_and_get_onnx_model,
                    generate_onnx_representation)


class TestFastT5(unittest.TestCase):
    model_name_or_path = 't5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    input = "translate English to French: The universe is a dark forest."
    tokenized_input = tokenizer(input, return_tensors='pt')
    expected_output = "L'univers est une forêt sombre."

    def test_translation(self):
        model = export_and_get_onnx_model(self.model_name_or_path)

        tokens = model.generate(input_ids=self.tokenized_input['input_ids'],
                                attention_mask=self.tokenized_input['attention_mask'], num_beams=2)

        output = self.tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)

        self.assertEqual(output, self.expected_output)

    def test_translation_without_quantization(self):
        model = export_and_get_onnx_model(self.model_name_or_path, quantized=False)

        tokens = model.generate(input_ids=self.tokenized_input['input_ids'],
                                attention_mask=self.tokenized_input['attention_mask'], num_beams=2)

        output = self.tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)

        self.assertEqual(output, self.expected_output)

    def test_custom_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            onnx_model_paths = generate_onnx_representation(self.model_name_or_path, output_path=tmpdirname)
            self.assertEqual(os.path.dirname(onnx_model_paths[0]), tmpdirname)
            self.assertEqual(len(os.listdir(tmpdirname)), 3)
