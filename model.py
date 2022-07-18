import torch
from layers.TransformerGraphEncoder import  TransformerGraphEncoder
from layers.SGCN import  SGCN
import torch.nn as nn
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.nn import functional as F
import torchmetrics
# import torch_optimizer as optim
# model definition




class STrGCN(pl.LightningModule):

    def __init__(self, adjacency_matrix,optimizer_params, num_classes : int=14,seq_len: int=8, d_model: int=512, n_heads: int=8,
                 nEncoderlayers: int=6, dropout: float = 0.1):
        super(STrGCN, self).__init__()
        # not the best model...

        features_in=3        
        
        self.Learning_Rate, self.betas, self.epsilon, self.weight_decay=optimizer_params
        self.num_classes=num_classes
        self.adjacency_matrix=adjacency_matrix.float()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.val_f1_score=torchmetrics.F1Score(num_classes)
        self.train_f1_score=torchmetrics.F1Score(num_classes)
        self.test_f1_score=torchmetrics.F1Score(num_classes)
        self.confusion_matrix=torchmetrics.ConfusionMatrix(num_classes)
        self.gcn=SGCN(features_in,d_model,self.adjacency_matrix)

        self.encoder=TransformerGraphEncoder(dropout=dropout,num_heads=n_heads,dim_model=d_model, num_layers=nEncoderlayers)

        self.out = nn.Sequential(
            nn.Linear(d_model, d_model,dtype=torch.float),
            nn.Mish(),
            # nn.Dropout(dropout),
            nn.LayerNorm(d_model,dtype=torch.float),
            nn.Linear(d_model,num_classes,dtype=torch.float)
          )

        self.d_model = d_model
        self.init_parameters()
    def init_parameters(self):
        for name,p in self.named_parameters() :
          if p.dim() > 1:
              nn.init.xavier_uniform_(p)
    def forward(self, x):
        x=x.type(torch.float)     

        #spatial features from SGCN
        x=self.gcn(x,self.adjacency_matrix)

        # temporal features from TGE
        x=self.encoder(x)

        # Global average pooling
        N,T,V,C=x.shape
        x=x.permute(0,3,1,2)
        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V)).view(N,C,T)
        # T pooling
        x = F.avg_pool1d(x, kernel_size=T).view(N,C)

        # Classifier
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

        #l1 regularization
        l1_lambda = 1e-4
        l1_norm = sum( p.abs().sum()  for p in self.parameters())

        loss_with_l1 = loss + l1_lambda * l1_norm

        self.train_acc(y_hat, y)
        self.train_f1_score(y_hat, y)
        self.log('train_loss', loss,on_epoch=True,on_step=True)
        self.log('train_acc', self.train_acc.compute(), prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_F1_score', self.train_f1_score.compute(), prog_bar=True, on_step=True, on_epoch=True)
        return loss_with_l1

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        
        x = batch["skeleton"].float()
        y = batch["label"]
        y = y.type(torch.LongTensor)
        y = y.cuda()
        targets = Variable(y, requires_grad=False)
        y_hat = self(x)
        
        loss = F.cross_entropy(y_hat, targets)

        self.valid_acc(y_hat, y)
        self.val_f1_score(y_hat, y)
        self.log('val_loss', loss, prog_bar=True,on_epoch=True,on_step=True)
        self.log('val_accuracy', self.valid_acc.compute(), prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_F1_score', self.val_f1_score.compute(), prog_bar=True, on_step=True, on_epoch=True)

    def training_epoch_end(self, outputs):
        #for name,p in self.named_parameters() :
        #    print(p.shape)
        
        self.train_acc.reset()

    def validation_epoch_end(self, outputs):
        self.valid_acc.reset()

    def test_step(self, batch, batch_nb):
        global confusion_matrix
        # OPTIONAL
        x = batch["skeleton"].float()
        y = batch["label"]
        y = y.type(torch.LongTensor)
        y = y.cuda()
        targets = Variable(y, requires_grad=False)
        y_hat = self(x)
        _, preds = torch.max(y_hat, 1)
        self.test_acc(y_hat, targets)
        self.test_f1_score(y_hat, y)
        loss = F.cross_entropy(y_hat, targets)        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', self.test_acc.compute(), prog_bar=True)
        self.log('test_F1_score', self.val_f1_score.compute(), prog_bar=True)
        
        confusion_matrix+=self.confusion_matrix(preds,targets)


    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        

        opt = torch.optim.RAdam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.Learning_Rate, weight_decay=self.weight_decay)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='min',
            factor=.5,
            patience=2,
            min_lr=1e-4,
            verbose=True
        )

        return  {"optimizer": opt, "lr_scheduler": reduce_lr_on_plateau, "monitor": "val_loss"}