import torch
from transformers import (
    T5Tokenizer,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
)


class DecoderWithLMhead(torch.nn.Module):
    """ Creation of a class to combine the decoder and the lm head """

    def __init__(self, decoder, lm_head, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config

    def forward(self, *inputs):

        input_ids, attention_mask, encoder_hidden_states = inputs[:3]

        list_pkv = inputs[3:]
        past_key_values = tuple(list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4))

        decoder_output = self.decoder(
            input_ids=input_ids,  # decoder_input_ids
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
        )

        lm_head_out = self.lm_head(decoder_output[0] * (self.config.d_model ** -0.5))

        return lm_head_out, decoder_output[1]


class T5Encoder(torch.nn.Module):
    """ Creation of a class to output only the last hidden state from the encoder """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, *input, **kwargs):
        return self.encoder(*input, **kwargs)[0]


class DecoderWithLMheadInitial(torch.nn.Module):
    """ Creation of a class to combine the decoder and the lm head """

    def __init__(self, decoder, lm_head, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config

    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        decoder_output = self.decoder(
            input_ids=input_ids,
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        return (
            self.lm_head(decoder_output[0] * (self.config.d_model ** -0.5)),
            decoder_output[1],
        )
