# Dual Triangle Attention - Positional aware bidirectional attention

In _Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings_, Gelberg, et al. from Sakana.ai present DroPE, a framework where natural language transformers are trained with rotary positional embeddings for some portion of training, and then the position embeddings are removed. Because vanilla transformers for NLP are typically autoregressive in nature, in other words, causal or unidirectional attention, the model can still intuit positional information by extracting how many tokens one token is attending to. The authors call this "exploitation" of the causal mask, which is accurate. Interestingly, after the removal of explicit positional information, the model learns to conduct language modeling tasks well without it, and its context is naturally extendable to longer than the training length. This is a big win for long context modeling!

However, in cases where bidirectional attention is ideal, this is not possible with naive implementations. Bidirectional attention treats every ij index of the attention matrix the same as ji, meaning the model cannot intuit where each token is in the matrix. It is completely position invariant, which is why positional information was injected into transformers in the first place. But bidirectional attention is a big advantage for modeling tasks like biological sequences. The reconstructing information at position i < j is not always necessary for text generation, but for protein structure or function prediction token j can very much influence i and i j.

To this end we invented Dual Triangle Attention, which splits the queries and keys in half, splitting along the hidden dimensions. By splitting the matrices in two, we calculate the upper triangle and lower triangle sections of the attention matrix with different componenents, allowing the model to view both ij and ji but in nuanced ways. This also reduces the FLOPs of the attention operation by 2 if implemented efficiently. We construct two experiments to explore position embedding free transformers in causal and bidirectional settings

Experiment 1:

We look at a toy objective that requires positional information. The task is essentially argmax, where we input random input ids and have the model predict which position the largest input_id resides. Naive bidirection models perform randomly at this objective, with causal and dual triangle modeling this task well. We sweep over many combinations of hidden size and number of layers for transformer models to intuit positional information. We try this object with learned postional embeddings, rotary embeddings, and the associated DrOPE strategies. This allows us to prove that dual triangle attention can model positional information in a bidirectional way, and allows us to track when this phenomena is emergent in model size.

Experiment 2:

We conduct a masked language modeling (MLM) objective for all combinations of positional embedding for causal and bidirectional variants. Yes, causal language modeling is not typcially used for MLM, but we do this to show the advantage of bidirectional variants for specific tasks. At a fixed number of layers of 12 and hidden size of 768, we train a fixed number of steps on a natural language dataset and then on a protein dataset with a set max length of 128. Then, we conduct inference on the test set with a max length of 512, and see which model combinations have the best naturally extended context window. We measure sequence reconstruction metrics, like language modeling loss (cross entropy), accuracy, f1, and mcc.

## Quickstart (Windows)

Create a virtual environment and install dependencies:

```powershell
git clone https://github.com/lhallee/position_encoding_exp.git
cd position_encoding_exp
chmod +x setup_env.sh
./setup_env.sh
source ~/benv/bin/activate
```

Run Experiment 1 (positional probe sweep):

```powershell
py experiment1.py
```

Run Experiment 2 (MLM on natural language + proteins):

```powershell
py experiment2.py
```

Experiment 2 runs **two independent sweeps** (NLP and protein), each with its own outputs/plots. It streams data from Hugging Face. The natural language dataset (FineWeb-Edu) is a very large stream, so the script creates small validation/test subsets by shuffling a streaming iterator and then taking a fixed number of documents. The protein dataset (Synthyra/omg_prot50) already includes train/valid/test splits.

## Outputs

Each experiment writes `results.csv` plus publication-ready figures in a `plots/` subfolder within the output directory. Experiment 2 produces:

- Training MLM loss over time for each attention/positional variant
- Validation reconstruction metrics (loss, accuracy, F1, MCC) over time
- Final extended-context test performance at length 512