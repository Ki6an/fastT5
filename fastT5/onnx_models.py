from .huggingface_utils import get_auth_token
from .ort_settings import get_onnx_runtime_sessions
from .onnx_exporter import (
    generate_onnx_representation,
    quantize,
    get_model_paths,
    saved_models_path,
)

from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
)
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    Seq2SeqLMOutput,
    BaseModelOutput,
)
import torch
import functools
import operator


class T5Encoder(torch.nn.Module):
    def __init__(self, encoder_sess):
        super().__init__()
        self.encoder = encoder_sess

    def forward(
        self,
        input_ids,
        attention_mask,
        inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        encoder_hidden_state = torch.from_numpy(
            self.encoder.run(
                None,
                {
                    "input_ids": input_ids.cpu().numpy(),
                    "attention_mask": attention_mask.cpu().numpy(),
                },
            )[0]
        )

        return BaseModelOutput(encoder_hidden_state)


class T5DecoderInit(torch.nn.Module):
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states):

        decoder_outputs = self.decoder.run(
            None,
            {
                "input_ids": input_ids.cpu().numpy(),
                "encoder_attention_mask": encoder_attention_mask.cpu().numpy(),
                "encoder_hidden_states": encoder_hidden_states.cpu().numpy(),
            },
        )

        list_pkv = tuple(torch.from_numpy(x) for x in decoder_outputs[1:])

        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return torch.from_numpy(decoder_outputs[0]), out_past_key_values


class T5Decoder(torch.nn.Module):
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, attention_mask, encoder_output, past_key_values):

        decoder_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "encoder_attention_mask": attention_mask.cpu().numpy(),
            "encoder_hidden_states": encoder_output.cpu().numpy(),
        }

        flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])

        past_key_values = {
            f"pkv_{i}": pkv.cpu().numpy() for i, pkv in enumerate(flat_past_key_values)
        }

        decoder_outputs = self.decoder.run(None, {**decoder_inputs, **past_key_values})
        # converts each value of the list to tensor from numpy
        list_pkv = tuple(torch.from_numpy(x) for x in decoder_outputs[1:])

        # creates a tuple of tuples of shape 6x4 from the above tuple
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return torch.from_numpy(decoder_outputs[0]), out_past_key_values


# config = T5Config.from_pretrained(model_or_model_path)


class OnnxT5(T5ForConditionalGeneration):
    """ creates a T5 model using onnx sessions (encode, decoder & init_decoder) """

    def __init__(self, model_or_model_path, onnx_model_sessions):
        config = T5Config.from_pretrained(model_or_model_path, use_auth_token=get_auth_token())
        super().__init__(config)

        assert len(onnx_model_sessions) == 3, "all three models should be given"

        encoder_sess, decoder_sess, decoder_sess_init = onnx_model_sessions

        self.encoder = T5Encoder(encoder_sess)
        self.decoder = T5Decoder(decoder_sess)
        self.decoder_init = T5DecoderInit(decoder_sess_init)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        encoder_hidden_states = encoder_outputs[0]

        if past_key_values is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if past_key_values is None:

            # runs only for the first time:
            init_onnx_outputs = self.decoder_init(
                decoder_input_ids, attention_mask, encoder_hidden_states
            )

            logits, past_key_values = init_onnx_outputs

        else:

            onnx_outputs = self.decoder(
                decoder_input_ids,
                attention_mask,
                encoder_hidden_states,
                past_key_values,
            )

            logits, past_key_values = onnx_outputs

        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)


def export_and_get_onnx_model(model_or_model_path, quantized=True):
    """
                          Method for whole pipeline,
    converts from pytorch to onnx --> quantizes model --> sets onnx runtime
                --> builds whole onnx model with all sessions

    """

    # Step 1. convert huggingfaces t5 model to onnx
    onnx_model_paths = generate_onnx_representation(model_or_model_path)

    if quantized:
        # Step 2. (recommended) quantize the converted model for fast inference and to reduce model size.
        quant_model_paths = quantize(onnx_model_paths)

        # step 3. setup onnx runtime
        print("Setting up onnx model...")
        model_sessions = get_onnx_runtime_sessions(quant_model_paths)
    else:
        print("Setting up onnx model...")
        model_sessions = get_onnx_runtime_sessions(onnx_model_paths)

    # step 4. get the onnx model
    model = OnnxT5(model_or_model_path, model_sessions)
    print("Done!")

    return model


def get_onnx_model(model_name_or_path, quantized=True):
    """ method gets the onnx model, if already converted models exists in models folder """

    encoder_path, decoder_path, init_decoder_path = get_model_paths(
        model_name_or_path, saved_models_path, quantized
    )

    if quantized:
        assert (
            encoder_path.exists()
            and decoder_path.exists()
            and init_decoder_path.exists()
        ), "quantized model don't exist in the model folder, first quantize the model!"
    else:
        assert (
            encoder_path.exists()
            and decoder_path.exists()
            and init_decoder_path.exists()
        ), "all or some models don't exists in the model folder, first convert the model! "

    model_paths = encoder_path, decoder_path, init_decoder_path

    model_sessions = get_onnx_runtime_sessions(model_paths)

    model = OnnxT5(model_name_or_path, model_sessions)

    return model
