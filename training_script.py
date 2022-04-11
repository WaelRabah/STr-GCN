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


def init_model(data_cfg, d_model,n_heads, adjacency_matrix,optimizer_params,dropout_rate=.1):
    if data_cfg == 0:
        num_classes = 14
    elif data_cfg == 1:
        num_classes = 28

    model = GCN_Transformer_Model(adjacency_matrix,optimizer_params,d_model=d_model,n_heads=n_heads,num_classes=num_classes, dropout=dropout_rate)

    return model


d_model=128
n_heads=4
batch_size = 32
n_training_steps=2660 // batch_size
warmup_steps=n_training_steps * 2
Learning_Rate = 1e-4
betas=(.9,.98)
epsilon=1e-9
optimizer_params=(Learning_Rate,betas,epsilon)
sequence_len = 8
workers = 4
test_subjects_ids = [1]
data_cfg = 0
Max_Epochs = 500
Early_Stopping = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = 3
dropout_rate=.3
cross_subject_validation_history = []
for iter in range(1, 2):
    print(f"Training Model N°{iter}")
# loading data
    train_loader, test_loader, val_loader, adjacency_matrix = init_data_loader(
        [iter], data_cfg, sequence_len, batch_size, workers, device)

    # folder for saving trained model...
    # change this path to the fold where you want to save your pre-trained model
    model_fold = "./TRANSFORMER+GCN-{}_dp-{}_lr-{}_dc-{}/".format(
        0, dropout_rate, Learning_Rate, data_cfg)
    try:
        os.mkdir(model_fold)
    except:
        pass

    # .........inital model
    print("\ninit model.............")
    model = init_model(data_cfg, d_model,n_heads, adjacency_matrix, optimizer_params,dropout_rate)
    print(f"d_model (the number of expected features in the encoder inputs/outputs):{d_model}")
    print(f"Number of heads :{n_heads}")
    print(f"Learning rate {Learning_Rate}")
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    early_stop_callback = EarlyStopping(
        monitor="val_accuracy", min_delta=0.00000001, patience=Early_Stopping, verbose=True, mode="max",check_on_train_epoch_end=True)
    logger = TensorBoardLogger("tb_logs", name=f"GCN_Model_{iter}")
    history=History_dict()
    trainer = pl.Trainer(gpus=1, precision=16, limit_train_batches=0.5, log_every_n_steps=1,
                         max_epochs=500, logger=[logger,history], callbacks=[early_stop_callback])

    # ***********training#***********
    trainer.fit(model, train_loader, val_loader)
    # cross_subject_validation_history.append(history)
    # max_accuracies = torch.FloatTensor(
    #     [hist["max_val_acc"] for hist in cross_subject_validation_history])
    # average_accuracy = max_accuracies.mean()-max_accuracies.std()
    # print(f"Average accuracy of the model over {iter} runs", max_accuracies)
    torch.cuda.empty_cache()



print(trainer.test(dataloaders=test_loader))

plt.figure(figsize=(15,8))
plot_history(history.history,"TRANSFORMER+GCN")