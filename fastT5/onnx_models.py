from .huggingface_utils import get_auth_token
from .ort_settings import get_onnx_runtime_sessions
from .onnx_exporter import (
    generate_onnx_representation,
    quantize,
    get_model_paths,
    saved_models_path,
)
from pathlib import Path

from transformers import (
    AutoConfig,
    MT5Config,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
    BaseModelOutput,
)
import torch
import functools
import operator
import numpy as np


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
        device = input_ids.device.type
        io_binding = self.encoder.io_binding()
        io_binding.bind_input(name="input_ids",
                                device_type=device,
                                device_id=input_ids.device.index if input_ids.device.index else 0,
                                element_type=np.longlong,
                                shape=list(input_ids.shape),
                                buffer_ptr=input_ids.data_ptr())
        io_binding.bind_input(name="attention_mask",
                                device_type=device,
                                device_id=attention_mask.device.index if attention_mask.device.index else 0,
                                element_type=np.longlong,
                                shape=list(attention_mask.shape),
                                buffer_ptr=attention_mask.data_ptr()) 
        io_binding.bind_output("hidden_states", input_ids.device.type)                                                             
        self.encoder.run_with_iobinding(io_binding)                                
        ort_output = io_binding.get_outputs()[0]

        return BaseModelOutput(ort_output)


class T5DecoderInit(torch.nn.Module):
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states):
        device = input_ids.device.type
        io_binding = self.decoder.io_binding()
        io_binding.bind_input(name="input_ids",
                                device_type=device,
                                device_id=input_ids.device.index if input_ids.device.index else 0,
                                element_type=np.longlong,
                                shape=list(input_ids.shape),
                                buffer_ptr=input_ids.data_ptr())
        io_binding.bind_input(name="encoder_attention_mask",
                                device_type=device,
                                device_id=encoder_attention_mask.device.index if encoder_attention_mask.device.index else 0,
                                element_type=np.longlong,
                                shape=list(encoder_attention_mask.shape),
                                buffer_ptr=encoder_attention_mask.data_ptr())
                              
        io_binding.bind_ortvalue_input("encoder_hidden_states", encoder_hidden_states)
        io_binding.bind_output("logits", device)
        io_binding.bind_output("past_key_values", device)

        for arg in self.decoder.get_outputs():
            io_binding.bind_output(arg.name, device)

        self.decoder.run_with_iobinding(io_binding)
        ort_output = io_binding.get_outputs()
        logits = ort_output[0]

        list_pkv = tuple(x for x in ort_output[1:])

        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return torch.from_numpy(logits.numpy()).to(device), out_past_key_values

class T5Decoder(torch.nn.Module):
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, attention_mask, encoder_output, past_key_values):
        device = input_ids.device.type
        io_binding = self.decoder.io_binding()
        io_binding.bind_input(name="input_ids",
                                device_type=device,
                                device_id=input_ids.device.index if input_ids.device.index else 0,
                                element_type=np.longlong,
                                shape=list(input_ids.shape),
                                buffer_ptr=input_ids.data_ptr())
        io_binding.bind_input(name="encoder_attention_mask",
                                device_type=device,
                                device_id=attention_mask.device.index if attention_mask.device.index else 0,
                                element_type=np.longlong,
                                shape=list(attention_mask.shape),
                                buffer_ptr=attention_mask.data_ptr())
                              
        io_binding.bind_ortvalue_input("encoder_hidden_states", encoder_output)

        flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])

        past_key_values = [
            (f"pkv_{i}", pkv) for i, pkv in enumerate(flat_past_key_values)
        ]
        
        for pkv in past_key_values:
            io_binding.bind_ortvalue_input(pkv[0], pkv[1])
        for arg in self.decoder.get_outputs():
            io_binding.bind_output(arg.name, device)

        self.decoder.run_with_iobinding(io_binding)
        ort_output = io_binding.get_outputs()
        logits = ort_output[0]

        list_pkv = tuple(x for x in ort_output[1:])

        # creates a tuple of tuples of shape 6x4 from the above tuple
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        # values of logits are not directly accessible. The workaround implies creating a new Tensor from
        # the numpy representation. A direct way to forward the Tensor would increase speed.
        return torch.from_numpy(logits.numpy()).to(device), out_past_key_values


class OnnxT5(T5ForConditionalGeneration):
    """creates a T5 model using onnx sessions (encode, decoder & init_decoder)"""

    def __init__(self, model_or_model_path, onnx_model_sessions):
        config = AutoConfig.from_pretrained(
            model_or_model_path, use_auth_token=get_auth_token()
        )
        super().__init__(config)

        # monkeypatch to work for MT5
        if (
            isinstance(model_or_model_path, str)
            and "mt5" in model_or_model_path.lower()
        ) or (
            hasattr(model_or_model_path, "name_or_path")
            and "mt5" in model_or_model_path.name_or_path
        ):
            self.model_type = "mt5"
            self.config_class = MT5Config
            self._keys_to_ignore_on_load_missing = [
                r"encoder\.embed_tokens\.weight",
            ]
            self._keys_to_ignore_on_save = [
                r"encoder\.embed_tokens\.weight",
            ]

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
        cross_attn_head_mask=None,
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


def export_and_get_onnx_model(
    model_or_model_path,
    custom_output_path=saved_models_path,
    quantized=True,
    input_sequence_length=256
):
    """
                          Method for whole pipeline,
    converts from pytorch to onnx --> quantizes model --> sets onnx runtime
                --> builds whole onnx model with all sessions

    """

    # Step 1. convert huggingfaces t5 model to onnx
    onnx_model_paths = generate_onnx_representation(
        model_or_model_path,
        output_path=custom_output_path,
        input_sequence_length=input_sequence_length
    )

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


def get_onnx_model(
    model_name_or_path, onnx_models_path=saved_models_path, quantized=True
):
    """
    method gets the onnx model, if already converted models exists
    Example:
    >> get_onnx_model(model_name_or_path="t5-finetuned", onnx_models_path="../models/onnx/quantized/")

    """

    encoder_path, decoder_path, init_decoder_path = get_model_paths(
        model_name_or_path, Path(onnx_models_path), quantized
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
