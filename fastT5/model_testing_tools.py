from time import perf_counter as pc
from matplotlib import pyplot as plt
from transformers import AutoTokenizer

import numpy as np


def speed_test(
    onnx_model,
    torch_model,
    beam_range: range = range(1, 10, 1),
    seq_length_range: range = range(10, 500, 50),
    input_text=None,
):
    """
        method prints the time took for onnx and pytorch model to finish a text generation task

    args:
        input_text (str) : text input for the model.
        onnx_model : onnx representation of the t5 model,
        torch_model : torch represention of the t5 model,
        beam_range (range) : provide a range, which takes starting end and steps (don't start with 0)
        sequence_length-range (range) : takes the start, end and steps as a range (start with 10)
    return :
        onnx_model_latency : numpy array of latency for each beam number and sequence length
        pytorch_model_latency : numpy array of latency for each beam number and sequence length
    """

    if input_text is None:
        input_text = """translate English to French: A nucleus is a collection of a large number of up and down quarks, confined into triplets (neutrons and protons). According to the strange matter hypothesis, strangelets are more stable than nuclei, so nuclei are expected to decay into strangelets. But this process may be extremely slow because there is a large energy barrier to overcome: 
                        as the weak interaction starts making a nucleus into a strangelet, the first few strange quarks form strange baryons, such as the Lambda, which are heavy. Only if many conversions occur almost simultaneously will the number of strange quarks reach the critical proportion required to achieve a lower energy state. This is very unlikely to happen, so even if the strange matter hypothesis were correct, nuclei would never be seen to decay to strangelets because their lifetime would be longer than the age of the universe.
                        The stability of strangelets depends on their size. This is because of (a) surface tension at the interface between quark matter and vacuum (which affects small strangelets more than big ones), and (b) screening of charges, which allows small strangelets to be charged, with a neutralizing cloud of electrons/positrons around them, but requires large strangelets, like any large piece of matter, to be electrically neutral in their interior. The charge screening distance tends to be of the order of a few femtometers, so only the outer few femtometers of a strangelet can carry charge.
                        The surface tension of strange matter is unknown. If it is smaller than a critical value (a few MeV per square femtometer) then large strangelets are unstable and will tend to fission into smaller strangelets (strange stars would still be stabilized by gravity). If it is larger than the critical value, then strangelets become more stable as they get bigger.
                        The known particles with strange quarks are unstable. Because the strange quark is heavier than the up and down quarks, it can spontaneously decay, via the weak interaction into an up quark. Consequently particles containing strange quarks, such as the Lambda particle, always lose their strangeness, by decaying into lighter particles containing only up and down quarks.
                        But condensed states with a larger number of quarks might not suffer from this instability. That possible stability against decay is the "strange matter hypothesis" proposed separately by Arnold Bodmer[3] and Edward Witten.[4] According to this hypothesis, when a large enough number of quarks are concentrated together, the lowest energy state is one which has roughly equal numbers of up, down, and strange quarks, namely a strangelet. This stability would occur because of the Pauli exclusion principle; having three types of quarks, rather than two as in normal nuclear matter, allows more quarks to be placed in lower energy levels
                     """

    tokenizer = AutoTokenizer.from_pretrained(torch_model.name_or_path)

    xx = []
    yy = []

    for j in beam_range:
        x = []
        y = []
        prev = [1, 2]
        for i in seq_length_range:

            token = tokenizer(
                input_text,
                padding=True,
                truncation=True,
                max_length=i,
                pad_to_max_length=i,
                return_tensors="pt",
            )

            input_ids = token["input_ids"]
            attention_mask = token["attention_mask"]

            a = pc()
            out = onnx_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=i,
                num_beams=j,
            )
            b = pc()
            x.append(b - a)

            c = pc()
            o = torch_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=i,
                num_beams=j,
            )
            d = pc()
            y.append(d - c)

            mean_y = np.mean(y)
            mean_x = np.mean(x)
            mean_ratio = mean_y / mean_x

            print(f"seqL : {i}, onnx-{b-a}, pt-{d-c} .. X faster {(d-c)/(b-a)}")

            # ...bleu_score-{bleu.compute(predictions=, references=[[tokenizer.decode(o.squeeze(), skip_special_tokens=True)], ])}')
            # print(f'o---{tokenizer.decode(out.squeeze(), skip_special_tokens=True)}...p---{tokenizer.decode(o.squeeze(), skip_special_tokens=True)}')

            if (o.shape[1] == prev[-1]) and (o.shape[1] == prev[-2]):
                break

            prev.append(o.shape[1])

        print(f"beam no.- {j} onnx-{mean_x} pt-{mean_y} X ratio-{mean_ratio}")

        xx.append(x)
        yy.append(y)
        plt.plot(x, "g", y, "r")
        plt.pause(0.05)

    plt.show()
    return np.array(xx), np.array(yy)
