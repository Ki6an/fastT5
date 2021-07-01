from .onnx_models_structure import (
    T5Encoder,
    DecoderWithLMhead,
    DecoderWithLMheadInitial,
)
from transformers import (
    T5Tokenizer,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
)
import torch
import functools
import operator
from progress.bar import Bar
from pathlib import Path
import os

_folder = Path.cwd()
saved_models_path = _folder.joinpath("models")

Bar.check_tty = False


def create_t5_encoder_decoder(pretrained_version="t5-base"):
    """Generates an encoder and a decoder model with a language model head from a pretrained huggingface model

    Args:
        pretrained_version (str): Name of a pretrained model, or path to a pretrained / finetuned version of T5

    Returns:
        simplified_encoder: pytorch t5 encoder with a wrapper to output only the hidden states
        decoder_with_lm_head: pytorch t5 decoder with a language modeling head
    """

    model = T5ForConditionalGeneration.from_pretrained(pretrained_version)

    return turn_model_into_encoder_decoder(model)


def turn_model_into_encoder_decoder(model):
    encoder = model.encoder
    decoder = model.decoder
    lm_head = model.lm_head

    decoder_with_lm_head = DecoderWithLMhead(decoder, lm_head, model.config)
    simplified_encoder = T5Encoder(encoder)
    decoder_with_lm_head_init = DecoderWithLMheadInitial(decoder, lm_head, model.config)

    return simplified_encoder, decoder_with_lm_head, decoder_with_lm_head_init


def generate_onnx_representation(pretrained_version=None, model=None, tokenizer_name_or_path=None, output_path=None):
    """Exports a given huggingface pretrained model, or a given model and tokenizer, to onnx

    Args:
        pretrained_version (str): Name of a pretrained model, or path to a pretrained / finetuned version of T5
        tokenizer_name_or_path (str): if missing then use pretrained_version
        output_path (str): if missing then use ./models
    """
    if (pretrained_version is None) and model is None:
        print(
            "You need to specify pretrained_version (the pretrained model you wish to export). Alternatively you can export a model you have in memory."
        )

        return
    if model is not None:
        (
            simplified_encoder,
            decoder_with_lm_head,
            decoder_with_lm_head_init,
        ) = turn_model_into_encoder_decoder(model)
    else:
        (
            simplified_encoder,
            decoder_with_lm_head,
            decoder_with_lm_head_init,
        ) = create_t5_encoder_decoder(pretrained_version)

    # model paths for enc, dec and dec_init
    output_path = saved_models_path if output_path is None else Path(output_path)
    encoder_path, decoder_path, init_decoder_path = get_model_paths(
        pretrained_version, output_path, quantized=False
    )

    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = pretrained_version
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    sample_input = "translate English to French: The universe is a dark forest."
    model_inputs = tokenizer(sample_input, return_tensors="pt")
    model_config = AutoConfig.from_pretrained(pretrained_version)

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    # dummy inputs
    batch_size = 5
    n_heads = model_config.num_heads
    seq_length_a, seq_length_b = input_ids.shape
    d_kv = model_config.d_kv

    input_ids_dec = torch.ones((5, 1), dtype=torch.int64)
    attention_mask_dec = torch.ones((5, seq_length_b), dtype=torch.int64)
    enc_out = torch.ones(
        (batch_size, seq_length_b, model_config.d_model), dtype=torch.float32
    )

    # self_attention_past_key_values = torch.ones(
    #     (model_config.num_decoder_layers, 2, batch_size, n_heads, seq_length_a, d_kv), dtype=torch.float32)
    # cross_attention_past_key_values = torch.ones(
    #     (model_config.num_decoder_layers, 2, batch_size, n_heads, seq_length_b, d_kv), dtype=torch.float32)

    sa = torch.ones(
        (batch_size, n_heads, seq_length_a, d_kv), dtype=torch.float32
    )  # 1, 8, 1, 64
    ca = torch.ones(
        (batch_size, n_heads, seq_length_b, d_kv), dtype=torch.float32
    )  # 1, 8, 30, 64
    t5_block = (sa, sa, ca, ca)
    past_key_values = (t5_block,) * model_config.num_decoder_layers

    flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])

    decoder_all_inputs = tuple(
        [input_ids_dec, attention_mask_dec, enc_out] + flat_past_key_values
    )

    num_of_inputs = 4 * model_config.num_decoder_layers

    # for progress bars
    bar = Bar("Exporting to onnx...", max=3)

    import warnings

    # ignores all the warnings during conversion
    warnings.filterwarnings("ignore")

    # Exports to ONNX
    with torch.no_grad():

        decoder_inputs = [
            "input_ids",
            "encoder_attention_mask",
            "encoder_hidden_states",
        ]

        pkv_input_names = ["pkv_{}".format(i) for i in range(0, num_of_inputs)]

        decoder_input_names = decoder_inputs + pkv_input_names

        decoder_output_names = ["logits", "output_past_key_values"]

        dyn_axis = {
            "input_ids": {0: "batch", 1: "sequence"},
            "encoder_attention_mask": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
            "output_past_key_values": {0: "batch", 1: "sequence"},
        }

        dyn_pkv = {
            "pkv_{}".format(i): {0: "batch", 2: "seq_length"}
            for i in range(0, num_of_inputs)
        }

        dyn_axis_params = {**dyn_axis, **dyn_pkv}

        # decoder to utilize past key values:
        torch.onnx.export(
            decoder_with_lm_head,
            decoder_all_inputs,
            decoder_path.as_posix(),
            export_params=True,
            do_constant_folding=True,
            opset_version=12,
            input_names=decoder_input_names,
            output_names=decoder_output_names,
            dynamic_axes=dyn_axis_params,
        )
        bar.next()

        torch.onnx.export(
            simplified_encoder,
            args=(input_ids, attention_mask),
            f=encoder_path.as_posix(),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "hidden_states": {0: "batch", 1: "sequence"},
            },
        )
        bar.next()
        # initial decoder to produce past key values
        torch.onnx.export(
            decoder_with_lm_head_init,
            (input_ids_dec, attention_mask_dec, enc_out),
            init_decoder_path.as_posix(),
            export_params=True,
            opset_version=12,
            input_names=[
                "input_ids",
                "encoder_attention_mask",
                "encoder_hidden_states",
            ],
            output_names=["logits", "past_key_values"],
            dynamic_axes={
                # batch_size, seq_length = input_shape
                "input_ids": {0: "batch", 1: "sequence"},
                "encoder_hidden_states": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
                "past_key_values": {0: "batch", 1: "sequence"},
                "encoder_attention_mask": {0: "batch", 1: "sequence"},
            },
        )
        bar.next()
        bar.finish()

    return encoder_path, decoder_path, init_decoder_path


def get_model_paths(pretrained_model, model_path, quantized):

    model_path.mkdir(parents=True, exist_ok=True)

    # gets only the filename
    pretrained_model_name = Path(pretrained_model).stem

    if not quantized:
        encoder_path = model_path.joinpath(f"{pretrained_model_name}-encoder.onnx")
        decoder_path = model_path.joinpath(f"{pretrained_model_name}-decoder.onnx")
        init_decoder_path = model_path.joinpath(
            f"{pretrained_model_name}-init-decoder.onnx"
        )
    else:
        encoder_path = model_path.joinpath(
            f"{pretrained_model_name}-encoder-quantized.onnx"
        )
        decoder_path = model_path.joinpath(
            f"{pretrained_model_name}-decoder-quantized.onnx"
        )
        init_decoder_path = model_path.joinpath(
            f"{pretrained_model_name}-init-decoder-quantized.onnx"
        )

    return encoder_path, decoder_path, init_decoder_path


def quantize(models_name_or_path):
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU
    Args:
        onnx_model_path: Path to location the exported ONNX model is stored
    Returns: The Path generated for the quantized
    """
    import onnx
    from onnxruntime.quantization import quantize, quantize_dynamic, QuantType

    bar = Bar("Quantizing...", max=3)

    quant_model_paths = []
    for model in models_name_or_path:
        model_name = model.as_posix()
        output_model_name = f"{model_name[:-5]}-quantized.onnx"
        quantize_dynamic(
            model_input=model_name,
            model_output=output_model_name,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
            optimize_model=False,
        )  # op_types_to_quantize=['MatMul', 'Relu', 'Add', 'Mul' ],
        quant_model_paths.append(output_model_name)
        bar.next()

    bar.finish()

    return tuple(quant_model_paths)
