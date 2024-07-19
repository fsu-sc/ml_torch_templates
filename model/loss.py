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
    # # Retrieve device from output tensor
    device = output.device
    n_components = dataloader.dataset.get_n_components()
    n_samples = len(output)
    
    temp_pca_components = nn.Parameter(dataloader.dataset.get_temp_pca_components().to(device), requires_grad=True)
    sal_pca_components = nn.Parameter(dataloader.dataset.get_sal_pca_components().to(device), requires_grad=True)

    # Split the predicted and true pcs for temp and sal
    pred_temp_pcs, pred_sal_pcs = output[:, :n_components], output[:, n_components:]
    true_temp_pcs, true_sal_pcs = target[:, :n_components], target[:, n_components:]
    
    # Inverse transform the PCA components to get the profiles
    pred_temp_profiles = torch.mm(pred_temp_pcs, temp_pca_components.T)
    pred_sal_profiles = torch.mm(pred_sal_pcs, sal_pca_components.T)
    true_temp_profiles = torch.mm(true_temp_pcs, temp_pca_components.T)
    true_sal_profiles = torch.mm(true_sal_pcs, sal_pca_components.T)
    
    # Calculate the MSE for temperature and salinity
    mse_temp = nn.functional.mse_loss(pred_temp_profiles, true_temp_profiles)
    mse_sal = nn.functional.mse_loss(pred_sal_profiles, true_sal_profiles)
    
    #calculate surface difference:
    pred_surface_t = pred_temp_profiles[:, 0]
    true_surface_t = nn.Parameter(dataloader.dataset.get_surface_T().to(device), requires_grad=True)
    pred_surface_s = pred_sal_profiles[:, 0]
    true_surface_s = nn.Parameter(dataloader.dataset.get_surface_S().to(device), requires_grad=True)
    # true_surface_s = true_sal_profiles[:, 0]
   
    #TODO: fix  
    # mse_surface_t = nn.functional.mse_loss(pred_surface_t, true_surface_t)
    # mse_surface_s = nn.functional.mse_loss(pred_surface_s, true_surface_s)
    # I'll add them as penalization terms
    
    # Weighted combination or simple averaging can be applied here
    #TODO: find better weighting, chose a better way to add penalization surface terms
    # return (mse_temp/(8**2) + mse_surface_t + mse_sal/(35**2) + mse_surface_s) / n_samples
    temp_range, sal_range = dataloader.dataset.get_range()
    return (mse_temp/(temp_range) + mse_sal/(sal_range)) / n_samples
    # return (mse_temp + mse_sal) / 2
    
def combined_loss(output, target, dataloader):
    return 5.5*PCA_loss(output, target, dataloader) + 3.6*weighted_mse_loss(output, target, dataloader)
    
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