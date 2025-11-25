from xmlrpc.client import Boolean
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch.nn import Module, Linear, Identity, ModuleList, Sequential, ReLU, Dropout
from torch_geometric.nn import TAGConv
from torch_geometric.data import Data
from typing import Optional, List, Tuple, Dict


class GraphMidpointJointTraining(Module):
    '''
    GraphMidpoint with optional early-exit joint training (Section 5.1).
    '''
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 input_time_dim: Optional[int] = None,
                 hidden_time_dim: Optional[int] = None,
                 time_aggr: Optional[str] = None,
                 readout: bool = True,
                 K: int = 2,
                 normalization: bool = True,
                 epsilon: float = 0.1,
                 activ_fun: Optional[str] = 'tanh',
                 use_previous_state: bool = False,
                 bias: bool = True,
                 exit_steps: Optional[List[int]] = None, # NEW: Early-exit parameters (Section 5.1)
                 loss_weights: Optional[List[float]] = None) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.K = K
        self.normalization = normalization
        self.activ_fun = getattr(torch, activ_fun) if activ_fun is not None else Identity()
        self.bias = bias
        self.input_time_dim = input_time_dim
        self.hidden_time_dim = hidden_time_dim
        self.use_previous_state = use_previous_state

        self.exit_steps = exit_steps if exit_steps is not None else []
        self.enable_early_exit = True 

        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim
    
        self.emb_t = None
        self.time_aggr = None
        if self.input_time_dim is not None:
            # Time encoder
            assert hidden_time_dim is not None, 'hidden_time_dim cannot be None when input_time_dim is not None'
            self.emb_t = Linear(self.input_time_dim, self.hidden_time_dim)
            
            assert time_aggr == 'concat' or time_aggr == 'add', f'time_aggr can be concat, add or None; not {time_aggr}'
            if time_aggr == 'concat':
                self.time_aggr = lambda x, y: torch.cat([x,y], dim=1)
                inp = inp + self.hidden_time_dim
            else:
                assert inp == self.hidden_time_dim
                self.time_aggr = lambda x, y: x+y

        self.conv = TAGConv(in_channels = inp,
                            out_channels = inp,
                            K = self.K,
                            normalize = self.normalization,
                            bias = self.bias)

        if not readout: 
            assert inp == self.output_dim, 'hidden_dim should be the same as output_dim when there is no readout'
        self.readout = Linear(inp, self.output_dim) if readout else None

        # Auxiliary classifiers for early exits (Eq. 3 from paper)
        self.auxiliary_heads = ModuleList([
            Sequential(
                Linear(inp, inp // 2),
                ReLU(),
                Dropout(0.1),
                Linear(inp // 2, self.output_dim)
            )
            for _ in self.exit_steps
        ])
            
        # Loss weights α_i (Eq. 6)
        if loss_weights is None:
            # Inception-style: 0.3 for auxiliary, 1.0 for final
            loss_weights = [0.3] * len(self.exit_steps) + [1.0]
        
        self.register_buffer(
            'loss_weights',
            torch.tensor(loss_weights, dtype=torch.float32)
        )

    def forward(self, 
                data: Data, 
                prev_h: Optional[torch.Tensor] = None,
                return_all_exits: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        """
        Forward pass with optional early exits.
        
        Args:
            data: PyG Data object
            prev_h: Previous hidden state (optional)
            return_all_exits: If True, return auxiliary predictions (for training)
        
        Returns:
            y: Final output [num_nodes, output_dim]
            h_middle: Final hidden state [num_nodes, hidden_dim]
            aux_outputs: List of (step, prediction) if return_all_exits=True, else None
        """
        x, edge_index, delta_t = data.x, data.edge_index, data.delta_t
        t_enc = data.t_enc if hasattr(data, 't_enc') else None

        # Build (node, timestamp) encoding
        h = self.emb(x) if self.emb else x
        if self.emb_t:
            t_enc = self.emb_t(t_enc) 
            h = self.time_aggr(h, t_enc)

        if self.use_previous_state and prev_h is not None: 
            h = h + prev_h

        # Storage for auxiliary outputs
        auxiliary_outputs = [] if return_all_exits else None

        # Midpoint integration loop
        for step in range(delta_t):
            conv = self.conv(h, edge_index)
            h_middle = h + 1/2 * self.epsilon * self.activ_fun(conv)
            conv_middle = self.conv(h_middle, edge_index)
            h = h + self.epsilon * self.activ_fun(conv_middle)

            # Check if this step is an exit point (Section 3.2)
            if return_all_exits and step in self.exit_steps:
                exit_idx = self.exit_steps.index(step)
                # Apply auxiliary classifier (Eq. 3)
                aux_pred = self.auxiliary_heads[exit_idx](h_middle)
                auxiliary_outputs.append((step, aux_pred))
        
        # Final output
        y = self.readout(h_middle) if self.readout is not None else h_middle
        
        # Add final prediction to auxiliary outputs
        auxiliary_outputs.append((delta_t, y))
        
        return y, h_middle, auxiliary_outputs

    # Joint loss computation (Eq. 6-7)
    def compute_joint_loss(self,
                          auxiliary_outputs: List[Tuple[int, torch.Tensor]],
                          targets: torch.Tensor,
                          criterion: Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute joint training loss from Section 5.1.
        
        Implements: L_total = L_final + Σ α_i * L_i  (Eq. 6)
        
        Args:
            auxiliary_outputs: List of (step, prediction) from forward()
            targets: Ground truth [num_nodes, output_dim]
            criterion: Loss function (e.g., MSELoss, CrossEntropyLoss)
        
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Breakdown for logging
        """        
        total_loss = 0.0
        loss_dict = {}
        
        for i, (step, pred) in enumerate(auxiliary_outputs):
            # Companion loss L_i (Eq. 7)
            loss_i = criterion(pred, targets)
            
            # Weight α_i (Eq. 6)
            weight = self.loss_weights[i]
            weighted_loss = weight * loss_i
            
            total_loss += weighted_loss
            
            # Logging
            loss_dict[f'loss_step_{step}'] = loss_i.item()
            loss_dict[f'weight_step_{step}'] = weight.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
