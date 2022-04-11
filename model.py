import torch
from layers import  Encoder, GCN
import torch.nn as nn
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.nn import functional as F
import torchmetrics
# model definition

class GCN_Transformer_Model(pl.LightningModule):

    def __init__(self, edge_index,optimizer_params, num_classes : int=14,time_len: int=8, d_model: int=512, n_heads: int=8,
                 nEncoderlayers: int=6, dropout: float = 0.5):
        super(GCN_Transformer_Model, self).__init__()
        # not the best model...

        head_dim = d_model // n_heads
        features_in=3
        gcn_out_size=3
        self.Learning_Rate, self.betas, self.epsilon=optimizer_params
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.graph_conv=GCN(features_in,gcn_out_size,dropout)
        self.gcn_embedding = nn.Sequential(
            nn.Linear(66, d_model,dtype=torch.double),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # self.pos_encoder=PositionalEncoding()
        self.edge_index=edge_index.type(torch.double)
        self.encoder=Encoder(nEncoderlayers,d_model, n_heads,head_dim)
        # self.encoder = nn.Embedding(time_len, d_model)
        self.d_model = d_model
        self.out=nn.Linear(d_model,num_classes,dtype=torch.double)
        self.init_parameters()
    def init_parameters(self):
        for name,p in self.named_parameters() :
          if p.dim() > 1:
              nn.init.xavier_uniform_(p)
    def forward(self, x):
        x=x.type(torch.double)     
        #skeleton to graph
        x=self.graph_conv(x,self.edge_index)
        # x=self.pos_encoder(x)
        x=torch.flatten(x,start_dim=-2)
        x=self.gcn_embedding(x)
        x=self.encoder(x)
        x=x.mean(dim=1)
        x=self.out(x)
        return x
    
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x = batch["skeleton"].float()
        y = batch["label"]
        y = y.type(torch.LongTensor)
        y = y.cuda()
        y = Variable(y, requires_grad=False)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log('train_loss', loss,on_epoch=True,on_step=True)
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x = batch["skeleton"].float()
        y = batch["label"]
        y = y.type(torch.LongTensor)
        y = y.cuda()
        y = Variable(y, requires_grad=False)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.valid_acc(y_hat, y)
        self.log('val_loss', loss, prog_bar=True,on_epoch=True,on_step=True)
        self.log('val_accuracy', self.valid_acc.compute(), prog_bar=True, on_step=True, on_epoch=True)
        return {"val_loss_step":loss,'val_accuracy_step':self.valid_acc.compute()}

    def training_epoch_end(self, outputs):
        # for name,p in self.named_parameters() :
        #   if name=="graph_conv.conv1.lin.weight":
        #     print(p)
        self.train_acc.reset()

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["val_loss_step"] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x["val_accuracy_step"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x = batch["skeleton"].float()
        y = batch["label"]
        y = y.type(torch.LongTensor)
        y = y.cuda()
        y = Variable(y, requires_grad=False)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.RAdam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.Learning_Rate,betas=self.betas,eps=self.epsilon)
        # return torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.Learning_Rate)