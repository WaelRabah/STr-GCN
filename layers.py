
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#layers definitions 
class TemporalGraphAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                head_num: int,
                head_dim: int
              ) -> None :
      super().__init__()
      self.head_dim=head_dim
      self.head_num=head_num
      self.d_model=d_model
      self.key_map = nn.Linear(d_model, head_dim * head_num,dtype=torch.double)
      self.query_map = nn.Linear(d_model, head_dim * head_num,dtype=torch.double)
      self.value_map = nn.Linear(d_model, head_dim * head_num,dtype=torch.double)
      self.out = nn.Linear(d_model, d_model,dtype=torch.double)
      self.init_parameters()
    def attention(self,Q,K,V):
      sqrt_dk=torch.sqrt(torch.tensor(self.head_dim))
      attention_weights=F.softmax((Q @ K.transpose(-2,-1))/sqrt_dk)
      attention_vectors=attention_weights @ V
      return attention_vectors
      
    def init_parameters(self):
      for p in self.parameters() :
          if p.dim() > 1:
              nn.init.xavier_uniform_(p)
  
    def forward(self,x: Tensor,adjacency_matrix : Tensor) -> torch.Tensor:
      batch_size = x.size(0)
      seq_length = x.size(1)
      graph_size=x.size(2)
      x=adjacency_matrix @ x
      K=self.key_map(x).view(batch_size, seq_length,graph_size, self.head_num, self.head_dim).transpose(1, 2)
      Q=self.query_map(x).view(batch_size, seq_length,graph_size, self.head_num, self.head_dim).transpose(1, 2)
      V=self.value_map(x).view(batch_size, seq_length,graph_size, self.head_num, self.head_dim).transpose(1, 2)
      x = self.attention(Q, K, V)
      x = x.transpose(1,2).contiguous().view(batch_size,seq_length,graph_size, self.d_model)
      x=self.out(x)
      return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

# class EncoderLayer(nn.Module):
#     def __init__(self,d_model, n_heads,head_dim, dropout = 0.1) -> None :
#       super().__init__()
#       self.att = nn.Sequential(
#         TemporalGraphAttention(d_model, n_heads,head_dim),
#         nn.Dropout(dropout)
#       )
#       self.fc = nn.Sequential(
#         nn.Linear(d_model,d_model,dtype=torch.double),
#         nn.ReLU(),
#         nn.Dropout(dropout)
#       )
#       self.norm=Norm(d_model)

#     def forward(self,x: Tensor) -> torch.Tensor:
#       x_att=self.att(x)
#       x=self.norm(x+x_att)
#       x_fc=self.fc(x)
#       x=self.norm(x+x_fc)
#       return x

# class Encoder(nn.Module):
#     def __init__(self,n_encoders,d_model, n_heads,head_dim, dropout = 0.1) -> None :
#       super().__init__()
#       self.encoders=nn.ModuleList([EncoderLayer(d_model,n_heads,head_dim)  for _ in range(n_encoders)])

#     def forward(self,x: Tensor) -> torch.Tensor:
#       for encoder in self.encoders :
#         x=encoder(x)
#       return x

class EncoderLayer(nn.Module):
    def __init__(self,d_model, n_heads,head_dim, dropout = 0.1) -> None :
      super().__init__()
      self.att = TemporalGraphAttention(d_model, n_heads,head_dim)
      self.fc = nn.Sequential(
        nn.Linear(d_model,d_model,dtype=torch.double),
        nn.ReLU()
      )
      self.norm=Norm(d_model)

    def forward(self,x: Tensor,adjacency_matrix : Tensor) -> torch.Tensor:
      x_att=self.att(x,adjacency_matrix)
      x=self.norm(x+x_att)
      x_fc=self.fc(x)
      x=self.norm(x+x_fc)
      return x

class Encoder(nn.Module):
    def __init__(self,n_encoders,d_model, n_heads,head_dim, dropout = 0.1) -> None :
      super().__init__()
      self.encoders=nn.ModuleList([EncoderLayer(d_model,n_heads,head_dim)  for _ in range(n_encoders)])
      self.post_encoder = nn.Sequential(
          nn.Linear(d_model, 16,dtype=torch.double),
          nn.ReLU(),
          nn.Dropout(dropout),
        )
      self.flatten_to_vec = nn.Sequential(
          nn.Linear(16*22*8, d_model,dtype=torch.double),
          nn.ReLU(),
          nn.Dropout(dropout),
        )
    def forward(self,x: Tensor,adjacency_matrix : Tensor) -> torch.Tensor:
      for encoder in self.encoders :
        x=encoder(x, adjacency_matrix)
      
      x=F.relu(self.post_encoder(x))
      x=torch.flatten(x,start_dim=1)
      x=self.flatten_to_vec(x)
      return x


class DenseGCNConv(torch.nn.Module):
    """See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.lin = nn.Linear(in_channels, out_channels, bias=False,dtype=torch.double)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)


    def forward(self, x, adj, mask=None, add_loop=True):
        """
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2
        out = self.lin(x)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

class PositionalEncoding(nn.Module):

    def __init__(self, max_len: int = 5000):
        super().__init__()
    def compute(self,sequence) : 
      def angle_between_pair(x,x_next):
        scalar_product=torch.einsum('jk,jk->j', x, x_next).unsqueeze(-1)
        l2_x=torch.sqrt(torch.einsum('jk,jk->j', x, x).unsqueeze(-1))
        l2_x_next=torch.sqrt(torch.einsum('jk,jk->j', x_next, x_next).unsqueeze(-1))
        angles=torch.arccos(scalar_product/(l2_x*l2_x_next))
        return torch.nan_to_num(angles) 
      distances=torch.tensor([],dtype=torch.float32,device=device)
      angles=torch.tensor([],dtype=torch.float32,device=device)
      for i in range(len(sequence)): 
        if i==len(sequence)-1 :
          x=sequence[i]
          x_next=sequence[i]
          l2_norm=torch.sqrt(torch.sum((x_next-x) ** 2,dim=1))
          distances=torch.cat((distances,l2_norm.unsqueeze(0)))
          angle_b_p=angle_between_pair(x,x_next)
          angles=torch.cat((angles,angle_b_p.unsqueeze(0)))
          continue
        x=sequence[i]
        x_next=sequence[i+1]
        l2_norm=torch.sqrt(torch.sum((x_next-x) ** 2,dim=1))
        distances=torch.cat((distances,l2_norm.unsqueeze(0)))
        angle_b_p=angle_between_pair(x,x_next)
        angles=torch.cat((angles,angle_b_p.unsqueeze(0)))
      return distances, angles

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [ batch_size, seq_len, n_nodes, n_features]
        """
        ones_vec=torch.tensor([1]).cuda()
        distances_per_sample=torch.tensor([],dtype=torch.float32).cuda()
        angles_per_sample=torch.tensor([],dtype=torch.float32).cuda()
        for sample in x:
          distances,angles=self.compute(sample)
          distances_per_sample=torch.cat((distances_per_sample,distances.unsqueeze(0)))
          angles_per_sample=torch.cat((angles_per_sample,angles.unsqueeze(0)))
        x_distance=(distances_per_sample.unsqueeze(-1)*ones_vec).type(torch.double)
        x_angle=(angles_per_sample*ones_vec).type(torch.double)
        x=torch.cat((x,Variable(x_distance, requires_grad=False).cuda()),dim=-1)
        x=torch.cat((x,Variable(x_angle, requires_grad=False).cuda()),dim=-1)
        return x

class GCN(nn.Module):
    def __init__(self,
               num_node_features: int,
               num_out: int,
               dp_rate : float
              ) -> None :
        super().__init__()
        self.dp_rate=dp_rate
        self.conv1=DenseGCNConv(num_node_features,8)
        self.conv2=DenseGCNConv(8,num_out)

    def forward(self,x: Tensor,adjacency_matrix: Tensor) -> torch.Tensor:
        x=self.conv1(x,adjacency_matrix)    
        x=F.relu(x)
        x=F.dropout(x,self.dp_rate)
        x=self.conv2(x,adjacency_matrix)
        x=F.relu(x)
        x=F.dropout(x,self.dp_rate)
        return x