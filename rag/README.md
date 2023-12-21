# RAG

## Script Execution

To run predictions for different RAG configurations, the rag_script shall be executed. The following parameters need to be considered:

- _batch_size_ (default &rarr; 8): Set the batch size for GPU inference to increase inference speed.
- variant (no default): Specify the name of the folder where the result files should be saved.
- _type_ (default &rarr; "validation"): Define on which data set the predictions should be made. Possible values are "train", "validation" and "test".
- _retriever_ (default &rarr; "hlatr"): Define which type of retriever shall be used: "langchain-vs" for FAISS retrieval (embeddings required), "bm25" or "hlatr".
- embeddings (default &rarr; "WhereIsAI/UAE-Large-V1"): Specify which type of embeddings from huggingface should be applied.
- _model_ (default &rarr; "google/flan-t5-base"): Specify which text2text LLM should be loaded to act as a generator. Enter the reference huggingface link.
- _format_text_ (toggle, default &rarr; false): Specify if basic preprocessing should be applied to source documents.
- _max_par_len_ (default &rarr; 1,000,000): Define the maximum length which paragraphs should not exceed. Default means no cutting.
- _topx_contexts_ (default &rarr; 1): Specify how many of the top x contexts shall be appended. Only can be used in combination with max_par_len. It is recommended to keep the maximum token input length of the model in mind when configuring these two parameters.
- _top_par_thresh_ (default &rarr; 0): Define whether there should be a threshold that defines the minimum score of a top-x candidate that needs to be met in order to be appended to the context. 0 means that there is no restriction.

An example of how an execution might look like is shown in the following example:

    python rag_script.py --variant mytest --model google/flan-t5-large --batch_size 64 --type test --retriever hlatr --topx_contexts 6 --max_par_len 50 --top_par_thresh 0 --format_text

## Model results

Most of the RAG pipeline results can be found [here](https://docs.google.com/spreadsheets/d/1qY-6oBOFPnoxMGfdVLLBO4X82lXMtANIh6Vc2xWeaCw/).
