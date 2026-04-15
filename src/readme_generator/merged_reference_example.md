# Running Llama-3.1-8B-Instruct on Xeon® with SGLang

## Model Acquisition

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

*Note:* You may need to log in your authorized HuggingFace account to access the model files.
Please refer to [HuggingFace login](https://huggingface.co/docs/huggingface_hub/quick-start#login).

## Model Deployment

The BKMs for model service deployment & benchmarking
for the supported data types are as follows:

[BF16](#bf16)

[W8A8_INT8](#w8a8_int8)

[FP8](#fp8)

[AWQ_INT4](#awq_int4)

### BF16

#### Environment Setup

`Llama-3.1-8B-Instruct` BF16 model has been supported by official SGLang.
Please refer to the `Installation` section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#installation).

#### Launch of the Serving Engine

An example command to launch SGLang server with `meta-llama/Llama-3.1-8B-Instruct` would be like:

```bash
sglang serve                     \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --enable-torch-compile       \
    --host 0.0.0.0               \
    --tp 6                       \
    --quantization <quantization>
```

The `<MODEL_ID_OR_PATH>` can be either the model ID (a.k.a. `meta-llama/Llama-3.1-8B-Instruct`)
or the path of the pre-downloaded model folder.

Please read the `Notes` part in the serving engine launching section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#launch-of-the-serving-engine)
to better undertand how to configure the arguments, especially for TP (tensor parallel)
and numa binding settings.

#### Benchmarking

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
the arguments of `bench_serving` are available via the command:

```bash
python -m sglang.bench_serving -h
```

### W8A8_INT8

#### Environment Setup

`Llama-3.1-8B-Instruct` W8A8_INT8 model has been supported by official SGLang.
Please refer to the `Installation` section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#installation).

#### Launch of the Serving Engine

An example command to launch SGLang server with `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` would be like:

```bash
sglang serve                     \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --quantization w8a8_int8     \
    --enable-torch-compile       \
    --host 0.0.0.0               \
    --tp 6                       \
    --quantization <quantization>
```

The `<MODEL_ID_OR_PATH>` can be either the model ID (a.k.a. `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8`)
or the path of the pre-downloaded model folder.

Please read the `Notes` part in the serving engine launching section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#launch-of-the-serving-engine)
to better undertand how to configure the arguments, especially for TP (tensor parallel)
and numa binding settings.

#### Benchmarking

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
the arguments of `bench_serving` are available via the command:

```bash
python -m sglang.bench_serving -h
```

### FP8

#### Environment Setup

`Llama-3.1-8B-Instruct` FP8 model has been supported by official SGLang.
Please refer to the `Installation` section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#installation).

#### Launch of the Serving Engine

An example command to launch SGLang server with `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8` would be like:

```bash
sglang serve                     \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --enable-torch-compile       \
    --host 0.0.0.0               \
    --tp 6                       \
    --quantization <quantization>
```

The `<MODEL_ID_OR_PATH>` can be either the model ID (a.k.a. `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`)
or the path of the pre-downloaded model folder.

Please read the `Notes` part in the serving engine launching section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#launch-of-the-serving-engine)
to better undertand how to configure the arguments, especially for TP (tensor parallel)
and numa binding settings.

#### Benchmarking

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
the arguments of `bench_serving` are available via the command:

```bash
python -m sglang.bench_serving -h
```

### AWQ_INT4

#### Environment Setup

`Llama-3.1-8B-Instruct` AWQ INT4 model is supported
in [a dev branch](https://github.com/jianan-gu/sglang/tree/cpu_optimized).

You can pull the docker image if you have access to `gar-registry.caas.intel.com`:

```bash
docker pull gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:intel-sglang-cpu-optimized
```

Or you can build the image from the Dockerfile:

```bash
git clone -b cpu_optimized https://github.com/jianan-gu/sglang.git
cd sglang/docker
# May need to add some other (e.g. proxy) settings
docker build -t sglang:intel-cpu-optimized -f xeon.Dockerfile .
```

#### Launch of the Serving Engine

An example command to launch SGLang server with `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` would be like:

```bash
python -m sglang.launch_server   \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --enable-torch-compile       \
    --host 0.0.0.0               \
    --tp 6                       \
    --quantization <quantization>
```

The `<MODEL_ID_OR_PATH>` can be either the model ID (a.k.a. `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`)
or the path of the pre-downloaded model folder.

Please read the `Notes` part in the serving engine launching section in
[the official SGLang CPU server document](https://docs.sglang.io/platforms/cpu_server.html#launch-of-the-serving-engine)
to better undertand how to configure the arguments, especially for TP (tensor parallel)
and numa binding settings.

#### Benchmarking

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
the arguments of `bench_serving` are available via the command:

```bash
python -m sglang.bench_serving -h
```