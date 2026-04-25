# Llama 3.1

## 1. Model Introduction

Llama 3.1 is a collection of pretrained and instruction tuned generative models, released in July 2024 by Meta. These models are available in 8B, 70B and 405B sizes, with the 405B variant being the most capable fully-open source model at the time.

These models bring open intelligence to all, with several new features and improvements:

- **Stronger General Intelligence**: These models showcase significant improvements in coding, state-of-the-art tool use, and overall stronger reasoning capabilities.
- **Extended Context Length**: Llama 3.1 extends the context length to 128K tokens to improve performance over long context tasks such as summarization and code reasoning.
- **Tool Use**: Llama 3.1 is trained to interact with a search engine, python interpreter and mathematical engine, and also improves zero-shot tool use capabilities to interact with potentially unseen tools.
- **Multilinguality**: Llama 3.1 supports 7 languages in addition to English: French, German, Hindi, Italian, Portuguese, Spanish, and Thai.

For further details, please refer to the [Llama 3.1 blog](https://ai.meta.com/blog/meta-llama-3-1/) and the [Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md).note

## 2. Model Acquisition

You can access the models on huggingface.

| Data Type | Model Card |
|:---:|:---:|
| BF16 | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| W8A8_INT8 | [RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8) |
| FP8 | [RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8) |
| AWQ_INT4 | [hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4) |

The models can be downloaded to local storage by command

```
hf download --resume <MODEL_ID> --local-dir 'path/to/local/dir'
```

Then set `--model-path <LOCAL_MODEL_PATH>` instead of `--model-path <MODEL_ID>`
in the `sglang.launch_server` command.

*Note:* You may need to log in your authorized HuggingFace account to access the model files.
Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

## 3. SGLang Installation

For BF16, W8A8_INT8 and FP8 data types,
please refer to the [official SGLang installation guide](https://docs.sglang.io/platforms/cpu_server.html#installation) for installation instructions.

For AWQ_INT4 data type, it is currently supported
in [a dev branch](https://github.com/jianan-gu/sglang/tree/cpu_optimized).

You can pull the docker image if you have access to `gar-registry.caas.intel.com`:

```bash
docker pull gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:intel-sglang-cpu-optimized
```

Or you can build the image from the Dockerfile:

```bash
git clone -b cpu_optimized https://github.com/jianan-gu/sglang.git
cd sglang/docker
# May need to add some other settings (e.g. proxy)
docker build -t sglang:intel-cpu-optimized -f xeon.Dockerfile .
```

## 4. Model Deployment

This section provides deployment configurations optimized for the hardware platforms and use cases.

**Interactive Command Generator**: Use the configuration selector below to generate a launch command for Llama 3.1 collection of models.

import Llama31ConfigGenerator from '@site/src/components/autoregressive/Llama31ConfigGenerator';

<Llama31ConfigGenerator />

Please read the `Notes` part in the serving engine launching section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#launch-of-the-serving-engine)
to better undertand how to configure the arguments, especially for TP (tensor parallel)
and numa binding settings.

## 5. Model Invocation

SGLang exposes an OpenAI-compatible endpoint. First, start the server

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="Meta-Llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function that retries a request with exponential backoff."},
    ],
    temperature=0.2,
    max_tokens=512,
)

print(resp.choices[0].message.content)
```

## 6. Benchmarking

Open another terminal and run the `sglang.bench_serving` command.
An example command would be like:

```bash
python -m sglang.bench_serving                                 \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json   \
    --dataset-name random                                      \
    --random-input-len 1024                                    \
    --random-output-len 1024                                   \
    --num-prompts 1                                            \
    --max-concurrency 1                                        \
    --request-rate inf                                         \
    --random-range-ratio 1.0
```

In the example command

- `--request_rate inf` indicates that all requests should be sent simultaneously.
- `--num-prompts 1` and `--max-concurrency 1` indicates 1 request is sent in this test round, can be adjusted for testing with different request concurrency number.
- `--dataset-name random` is set to randomly select samples from the dataset.
- `--random-input 1024`, `--random-output 1024` and `--random-range-ratio 1.0` settings are for fixed 1024-in/1024-out token size limit (realized by truncating or repeating the original sample).
 
Please adjust the settings per your benchmarking scenarios. Detailed descriptions for
the arguments of `bench_serving` are available
in [this doc](../../base/benchmarks/lm_benchmark.md).
