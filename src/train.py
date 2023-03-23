import argparse
import os
import torch
import torch.nn as nn
from src.utils.utils import (DATALOADER , Functional)
from src.trainer import Trainer
from src.models import( WORD2VECTCBOW ,WORD2VECTCBOWSKIPGRAM)

class Train():
    def __init__(self) -> None:
        pass

    def train(config):
        try:
            os.makedirs(config.get("hyperparameters","model_dir"))
        except FileExistsError:
            pass

        train_dataloader, vocab =DATALOADER().DataLoader(
            model_name=config.get("hyperparameters","model_name"),
            ds_name=config.get("hyperparameters","dataset"),
            ds_type="train",
            data_dir=config.get("hyperparameters","data_dir"),
            batch_size=int(config.get("hyperparameters","train_batch_size")),
            shuffle=config.get("hyperparameters","shuffle"),
            vocab=None,
        )

        val_dataloader, _ = DATALOADER().DataLoader(
            model_name=config.get("hyperparameters","model_name"),
            ds_name=config.get("hyperparameters","dataset"),
            ds_type="valid",
            data_dir=config.get("hyperparameters","data_dir"),
            batch_size=int(config.get("hyperparameters","val_batch_size")),
            shuffle=config.get("hyperparameters","shuffle"),
            vocab=vocab,
        )

        vocab_size = len(vocab.get_stoi())
        print(f"Vocabulary size: {vocab_size}")

        if config.get("hyperparameters","model_name") == "CBOW":
            model_class = WORD2VECTCBOW
        else:
            model_class = WORD2VECTCBOWSKIPGRAM

        model = model_class(vocab_size=vocab_size)
        criterion = nn.CrossEntropyLoss()

        optimizer_class = Functional().get_optimizer_class(name = config.get("hyperparameters","optimizer"))
        optimizer = optimizer_class(model.parameters(), lr=float(config.get("hyperparameters","learning_rate")))
        lr_scheduler = Functional().get_lr_scheduler(optimizer, int(config.get("hyperparameters","epochs")), verbose=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trainer = Trainer(
            model=model,
            epochs=int(config.get("hyperparameters","epochs")),
            train_dataloader=train_dataloader,
            train_steps=config.get("hyperparameters","train_steps"),
            val_dataloader=val_dataloader,
            val_steps=config.get("hyperparameters","val_steps"),
            criterion=criterion,
            optimizer=optimizer,
            checkpoint_frequency=config.get("hyperparameters","checkpoint_frequency"),
            lr_scheduler=lr_scheduler,
            device=device,
            model_dir=config.get("hyperparameters","model_dir"),
            model_name=config.get("hyperparameters","model_name"),
        )

        trainer.train()
        print("Training finished.")

        trainer.save_model()
        trainer.save_loss()
        Functional.save_vocab(vocab, config.get("hyperparameters","model_dir"))
        Functional.save_config(config, config.get("hyperparameters","model_dir"))
        print("Model artifacts saved to folder:", config.get("hyperparameters","model_dir"))


