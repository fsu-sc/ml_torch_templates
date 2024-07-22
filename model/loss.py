import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def nll_loss(output, target):
    return F.nll_loss(output, target)

def rmse_loss(output, target):
    return torch.sqrt(F.mse_loss(output, target))

def weighted_mse_loss(output, target, data_loader):
    # Retrieve device from output tensor
    device = output.device

    # Get weights from the data loader and move them to the correct device
    weights = data_loader.dataset.get_pca_variances().float().to(device)

    # Compute the weighted MSE loss
    squared_diff = (output - target) ** 2
    weighted_squared_diff = weights * squared_diff
    loss = weighted_squared_diff.mean()
    return loss

# TODO: fix multi-gpu implementation
def PCA_loss(output, target, dataloader):
    # Retrieve device from output tensor
    device = output.device
    n_components = dataloader.dataset.get_n_components()
    n_samples = len(output)
    
    temp_pca_components = nn.Parameter(dataloader.dataset.get_temp_pca_components().to(device), requires_grad=True)
    sal_pca_components = nn.Parameter(dataloader.dataset.get_sal_pca_components().to(device), requires_grad=True)

    # Split the predicted and true pcs for temp and sal
    pred_temp_pcs, pred_sal_pcs = output[:, :n_components], output[:, n_components:]
    true_temp_pcs, true_sal_pcs = target[:, :n_components], target[:, n_components:]
    
    # Inverse transform the PCA components to get the profiles
    pred_temp_profiles = torch.mm(pred_temp_pcs, temp_pca_components.T).to(device)
    pred_sal_profiles = torch.mm(pred_sal_pcs, sal_pca_components.T).to(device)
    true_temp_profiles = torch.mm(true_temp_pcs, temp_pca_components.T).to(device)
    true_sal_profiles = torch.mm(true_sal_pcs, sal_pca_components.T).to(device)
    
    # Calculate the MSE for temperature and salinity
    mse_temp = F.mse_loss(pred_temp_profiles, true_temp_profiles).to(device)
    mse_sal = F.mse_loss(pred_sal_profiles, true_sal_profiles).to(device)
    
    # Calculate surface difference
    pred_surface_t = pred_temp_profiles[:, 0]
    true_surface_t = nn.Parameter(dataloader.dataset.get_surface_T().to(device), requires_grad=True)
    pred_surface_s = pred_sal_profiles[:, 0]
    true_surface_s = nn.Parameter(dataloader.dataset.get_surface_S().to(device), requires_grad=True)
    
    # Calculate surface penalization terms (optional, commented out in this example)
    # mse_surface_t = F.mse_loss(pred_surface_t, true_surface_t)
    # mse_surface_s = F.mse_loss(pred_surface_s, true_surface_s)
    
    # Apply the correct weighting and range
    temp_range, sal_range = dataloader.dataset.get_range()
    temp_range = torch.tensor(temp_range, device=device) if isinstance(temp_range, torch.Tensor) else torch.tensor(temp_range).to(device)
    sal_range = torch.tensor(sal_range, device=device) if isinstance(sal_range, torch.Tensor) else torch.tensor(sal_range).to(device)

    # Calculate the final loss, ensuring `n_samples` is an integer
    A = mse_temp / temp_range
    B = mse_sal / sal_range
    return (A + B) / n_samples

def combined_loss(output, target, dataloader):
    return 5.5 * PCA_loss(output, target, dataloader) + 3.6 * weighted_mse_loss(output, target, dataloader)

    
# class CombinedPCALoss(nn.Module):
#     def __init__(self, temp_pca, sal_pca, n_components, weights, device):
#         super(CombinedPCALoss, self).__init__()
#         self.pca_loss = PCALoss(temp_pca, sal_pca, n_components)
#         self.weighted_mse_loss = genWeightedMSELoss(n_components, device, weights)

#     def forward(self, pcs, targets):
#         # Calculate the PCA loss
#         pca_loss = self.pca_loss(pcs, targets)

#         # Calculate the weighted MSE loss
#         weighted_mse_loss = self.weighted_mse_loss(pcs, targets)

#         # Combine the losses
#         # You may need to adjust the scaling factor to balance the two losses
#         combined_loss = 5.5*pca_loss + 3.6*weighted_mse_loss
#         return combined_loss