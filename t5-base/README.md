# T5 Fine-tuning

## Script Execution

### Fine-Tuning

To fine-tune a T5 model, the t5_pipeline script shall be executed. The following parameters need to be considered:

- _batch_size_ (default &rarr; 8): Set the batch size for GPU training to increase inference speed.
- _model_: Specify which T5 variant should be loaded to be fine-tuned. Enter the reference huggingface link or a local path.
- _tokenizer_: Specify which T5 variant should be loaded to act as a tokenizer. Enter the reference huggingface link or a local path.
- _domain_ (default &rarr; wikipedia): Specify the domain of the data to be fine-tuned on. The domain is used to load the correct dataset.
- _epochs_ (default &rarr; 5): Specify the number of epochs to train the model.

An example of how an execution might look like is shown in the following example:

    python t5_pipeline.py --batch_size 8 --model google/flan-t5-large --tokenizer google/flan-t5-large --domain wikipedia --epochs 5

The script will automatically save the model and tokenizer in the `models` folder. The model will be saved after each epoch and a prediction will be generated for the final model on the test set.

### Prediction

To generate predictions with a fine-tuned T5 model, the t5_predict script shall be executed. The following parameters need to be considered:

- _model_: Specify which T5 variant should be loaded to act as a generator. Enter the reference huggingface link or local path.
- _tokenizer_: Specify which T5 variant should be loaded to act as a tokenizer. Enter the reference huggingface link or a local path.
- _domain_ (default &rarr; wikipedia): Specify the domain of the data to be fine-tuned on. The domain is used to load the correct dataset.
- _retriever_ (default &rarr; None): Define which type of retriever shall be used: "langchain-vs" for FAISS retrieval (embeddings required), "bm25" or "hlatr".
- _type_ (default &rarr; "validation"): Define on which data set the predictions should be made. Possible values are "train", "validation" and "test".
- _format_text_ (toggle, default &rarr; false): Specify if basic preprocessing should be applied to source documents.

An example of how an execution might look like is shown in the following example:

    python t5_pipeline.py --model google/flan-t5-large --tokenizer google/flan-t5-large --domain wikipedia --retriever hlatr --type test --format_text

## Model results

Most of the T5 pipeline results can be found [here](https://docs.google.com/spreadsheets/d/1qY-6oBOFPnoxMGfdVLLBO4X82lXMtANIh6Vc2xWeaCw/).
