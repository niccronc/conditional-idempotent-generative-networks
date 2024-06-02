When starting a training job, the main.py function provides the following options:

1. -s (--seed). Random seed, integer, default=0.
2. -ld (--lateng-dim). Latent dimension, integer, default=16.
3. -id (--intermediate-dim). Dimension of the intermediate latent tensor, integer, default=512.
4. -ed (--embedding-dim). Dimension of the label embeddings, integer, default=10.
5. -lr (--learning-rate). Learning rate of the AdamW optimizer, float, default=0.0001.
6. -b (--batch-size). Batch size to use during training and validation, integer, default=256.
7. -e (--num-epochs). Number of training epochs, integer, default=100.
8. -w (--num-workers). Number of workers to use for the training job, integer, default=0 (single worker).
9. -c (--conditioning). Type of conditioning approach to use during trainind, string. Only 'channel' and 'filter' are currently supported, with default='channel'.
10. -d (--download). If passed, downloads the MNIST dataset from source, otherwise loads it from a cache folder (if available).
