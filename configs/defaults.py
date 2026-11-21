DEFAULTS = {
    "image_size": 224,
    "normalize": True,
    "pretrained": True,
    "epochs": 100,
    "batch_size": 64,
    "evaluation_batch_size": 64,
    "num_workers": 4,
    "drop_last": False,
    "deterministic": True,
    "checkpoint_monitor": "accuracy",
    "log_interval": 50,
    "scheduler": {
        "name": "cosine",
        "eta_min": 0.0,
    },
}
