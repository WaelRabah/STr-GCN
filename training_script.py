#!/usr/bin/env python


# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# pip install numpy matplotlib
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
# pip install wget
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from data_loader import init_data_loader
from model import GCN_Transformer_Model
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import collections
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
from ray.tune import CLIReporter
class History_dict(LightningLoggerBase):
    def __init__(self):
        super().__init__()

        self.history = collections.defaultdict(list) # copy not necessary here  
        # The defaultdict in contrast will simply create any items that you try to access

    @property
    def name(self):
        return "Logger_custom_plot"

    @property
    def version(self):
        return "1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for metric_name, metric_value in metrics.items():
            if metric_name != 'epoch':
                self.history[metric_name].append(metric_value)
            else: # case epoch. We want to avoid adding multiple times the same. It happens for multiple losses.
                if (not len(self.history['epoch']) or    # len == 0:
                    not self.history['epoch'][-1] == metric_value) : # the last values of epochs is not the one we are currently trying to add.
                    self.history['epoch'].append(metric_value)
                else:
                    pass
        return

    def log_hyperparams(self, params):
        pass


def plot_history(history, title: str) -> None:
    ax1 = plt.subplot(211)
    ax1.set_title("Loss")
    ax1.plot(history["train_loss_epoch"], label="train")
    ax1.plot(history["val_loss_epoch"], label="val")
    plt.xlabel("Epoch")
    ax1.legend()

    ax1 = plt.subplot(212)
    ax1.set_title("Accuracy")
    ax1.plot(history["train_acc_epoch"], label="train")
    ax1.plot(history["val_accuracy_epoch"], label="val")
    plt.xlabel("Epoch")
    ax1.legend()
    plt.tight_layout()
    plt.show()


def init_model(data_cfg, d_model, edge_index,optimizer_params):
    if data_cfg == 0:
        num_classes = 14
    elif data_cfg == 1:
        num_classes = 28

    model = GCN_Transformer_Model(edge_index,optimizer_params,d_model=d_model,num_classes=num_classes, dropout=.2)

    return model

def train(config):
    d_model=config['d_model']
    batch_size = config['batch_size']
    # n_training_steps=2660 // batch_size
    # warmup_steps=n_training_steps * 2
    Learning_Rate = config['lr']
    betas=(.9,.98)
    epsilon=1e-9
    optimizer_params=(Learning_Rate,betas,epsilon)
    sequence_len = 8
    workers = 4
    data_cfg = 0
    Max_Epochs = 500
    Early_Stopping = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_features = 3
    # cross_subject_validation_history = []
    for iter in range(1, 2):
        print(f"Training Model N°{iter}")
    # loading data
        train_loader, val_loader, edge_index = init_data_loader(
            [iter], data_cfg, sequence_len, batch_size, workers, device)

        # folder for saving trained model...
        # change this path to the fold where you want to save your pre-trained model
        model_fold = "./TRANSFORMER+GCN-{}_dp-{}_lr-{}_dc-{}/".format(
            0, .5, Learning_Rate, data_cfg)
        try:
            os.mkdir(model_fold)
        except:
            pass

        # .........inital model
        print("\ninit model.............")
        model = init_model(data_cfg, d_model, edge_index, optimizer_params)
        print(f"Learning rate {Learning_Rate}")
        metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
        tr_cb=TuneReportCallback(metrics, on="validation_end")
        early_stop_callback = EarlyStopping(
            monitor="val_accuracy", min_delta=0.00000001, patience=Early_Stopping, verbose=True, mode="max",check_on_train_epoch_end=True)
        logger = TensorBoardLogger("tb_logs", name=f"GCN+Transformer_Model_{iter}")
        history=History_dict()
        trainer = pl.Trainer(gpus=1, precision=16, limit_train_batches=0.5, log_every_n_steps=1,
                            max_epochs=Max_Epochs, logger=[logger,history], callbacks=[early_stop_callback, tr_cb])

        # ***********training#***********
        trainer.fit(model, train_loader, val_loader)
        # cross_subject_validation_history.append(history)
        # max_accuracies = torch.FloatTensor(
        #     [hist["max_val_acc"] for hist in cross_subject_validation_history])
        # average_accuracy = max_accuracies.mean()-max_accuracies.std()
        # print(f"Average accuracy of the model over {iter} runs", max_accuracies)
        torch.cuda.empty_cache()
        # plt.figure(figsize=(15,8))
        # plot_history(history.history,"TRANSFORMER+GCN")


config = {
 "d_model": tune.choice([ 64, 128, 256]),
 "lr": tune.choice([1e-4,1e-3,1e-2]),
 "batch_size": tune.choice([32])
}
reporter=CLIReporter(max_report_frequency=30)
trainable = tune.with_parameters(
    train)

analysis = tune.run(
    trainable,
    resources_per_trial={
        "gpu": 1
    },
    metric="acc",
    mode="max",
    config=config,
    name="tune_Transformer+GCN",
    progress_reporter=reporter,
    num_samples=12
    )

print(analysis.best_config)