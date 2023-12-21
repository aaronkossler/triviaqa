# HuggingFace Pipeline

## Script Execution

To run predictions for different HuggingFace models, the pre_pipeline script shall be executed. The following parameters need to be considered:

- _model_: The name of the model to be used. The model name can be found on the [HuggingFace website](https://huggingface.co/models).
- _domain_ (default &rarr; wikipedia): Specify the domain of the data to be fine-tuned on. The domain is used to load the correct dataset.
- _gpu_ (default &rarr; yes): Specify if GPU should be used for inference.

An example of how an execution might look like is shown in the following example:

    python pre_pipeline.py --model deepset/minilm-uncased-squad2 --domain wikipedia --gpu yes

## Model results

The results can be found in the project report.
