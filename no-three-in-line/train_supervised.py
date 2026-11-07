from __future__ import annotations

from supervised.config import load_supervised_config
from supervised.preprocess import load_preprocessed_cache
from supervised.trainer import SupervisedTrainer


def main() -> None:
    """Load cached supervised data and run the two-stage training curriculum."""

    config = load_supervised_config()
    train_records = load_preprocessed_cache(config.data_paths.train_cache)
    # train_records = train_records[2000:2000+config.runtime.batch_size*1]
    val_records = None
    if config.data_paths.val_cache:
        val_records = load_preprocessed_cache(config.data_paths.val_cache)

    trainer = SupervisedTrainer(config)
    trainer.fit(train_records=train_records, val_records=val_records)


if __name__ == "__main__":
    main()
