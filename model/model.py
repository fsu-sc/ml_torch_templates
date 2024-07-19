import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import torch.nn as nn

def activation_selector(activation_name):
    # This function maps activation names to their corresponding PyTorch modules
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'linear': nn.Identity()  # Using Identity for linear activation
    }
    return activations.get(activation_name.lower(), nn.Identity())  # Default to linear if not found

class NeSPReSO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_config=[512, 512], dropout_prob=0.2, hidden_activation='relu', last_activation='linear', num_heads=9):
        super().__init__()

        # Construct layers based on the given configuration
        layers = []
        prev_dim = input_dim
        for neurons in hidden_dim_config:
            layers.append(nn.Linear(prev_dim, neurons))
            activation = activation_selector(hidden_activation)
            
            layers.append(activation)  # Now using Module subclass
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))  # added dropout
            prev_dim = neurons
        layers.append(nn.Linear(prev_dim, output_dim))
        last_activation_module = activation_selector(last_activation)
        layers.append(last_activation_module)
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(SelfAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert (
#             self.head_dim * num_heads == embed_dim
#         ), "embed_dim must be divisible by num_heads"

#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.fc_out = nn.Linear(embed_dim, embed_dim)

#     def forward(self, x):
#         N, seq_length, embed_dim = x.shape

#         # Split the embedding into multiple heads for multi-head attention
#         queries = self.query(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
#         keys = self.key(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
#         values = self.value(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

#         energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
#         attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

#         out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_length, embed_dim)
#         out = self.fc_out(out)

#         return out, attention


# class NeSPReSO(BaseModel):
#     def __init__(self, input_dim, output_dim, hidden_dim_config=[512, 512], dropout_prob=0.2, hidden_activation='relu', last_activation='linear', num_heads=9):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.hidden_activation = activation_selector(hidden_activation)
#         self.last_activation = activation_selector(last_activation)
        
#         self.self_attention = SelfAttention(input_dim, num_heads)

#         prev_dim = input_dim
        
#         for neurons in hidden_dim_config:
#             self.layers.append(nn.Linear(prev_dim, neurons))
#             self.layers.append(nn.Dropout(dropout_prob))
#             prev_dim = neurons
        
#         self.final_fc = nn.Linear(prev_dim, output_dim)
        
#     def forward(self, x):
#         x, self.attention_weights = self.self_attention(x)
        
#         for i in range(0, len(self.layers), 2):
#             x = self.hidden_activation(self.layers[i](x))
#             x = self.layers[i + 1](x)  # Dropout
    
#         x = self.final_fc(x)
#         x = self.last_activation(x)
#         return x
