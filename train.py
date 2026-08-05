import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def masked_mse_loss(reconstruction: torch.Tensor, target: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
    """
    Computes MSE loss only on unmasked (valid) features.
    """
    valid_mask = (target != mask_value)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True).to(reconstruction.device)
    
    return F.mse_loss(reconstruction[valid_mask], target[valid_mask])


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    classification_criterion: nn.Module,
    device: torch.device,
    alpha: float,
    noise_factor: float,
) -> tuple[float, float, float]:
    """
    Handles a single epoch of training.
    Returns: (avg_joint_loss, avg_recon_loss, avg_class_loss)
    """
    model.train()
    total_train_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    
    for batch_prot, batch_phos, batch_y in train_loader:
        batch_prot = batch_prot.to(device)
        batch_phos = batch_phos.to(device)
        batch_y = batch_y.to(device)
        
        noise_prot = torch.randn_like(batch_prot) * noise_factor
        noisy_prot = batch_prot + (noise_prot * (batch_prot != 0.0).float())
        
        noise_phos = torch.randn_like(batch_phos) * noise_factor
        noisy_phos = batch_phos + (noise_phos * (batch_phos != 0.0).float())
        
        optimizer.zero_grad()
        
        recon_prot, recon_phos, fused_latent, logits = model(noisy_prot, noisy_phos)
        
        recon_loss_prot = masked_mse_loss(recon_prot, batch_prot)
        recon_loss_phos = masked_mse_loss(recon_phos, batch_phos)
        batch_recon_loss = recon_loss_prot + recon_loss_phos
        
        class_loss = classification_criterion(logits, batch_y.view_as(logits))
        joint_loss = batch_recon_loss + (alpha * class_loss)
        
        joint_loss.backward()
        optimizer.step()
        
        total_train_loss += joint_loss.item()
        total_recon_loss += batch_recon_loss.item()
        total_class_loss += class_loss.item()
        
    num_batches = len(train_loader)
    return total_train_loss / num_batches, total_recon_loss / num_batches, total_class_loss / num_batches


def _validate_model(
    model: nn.Module,
    test_loader: DataLoader,
    classification_criterion: nn.Module,
    device: torch.device,
    alpha: float,
) -> tuple[float, float, float]:
    """
    Handles validation loop.
    Returns: (avg_joint_loss, avg_recon_loss, avg_class_loss)
    """
    model.eval()
    val_loss = 0.0
    val_recon = 0.0
    val_class = 0.0
    
    with torch.no_grad():
        for batch_prot, batch_phos, batch_y in test_loader:
            batch_prot = batch_prot.to(device)
            batch_phos = batch_phos.to(device)
            batch_y = batch_y.to(device)
            
            recon_prot, recon_phos, fused_latent, logits = model(batch_prot, batch_phos)
            
            r_loss_prot = masked_mse_loss(recon_prot, batch_prot)
            r_loss_phos = masked_mse_loss(recon_phos, batch_phos)
            total_r_loss = r_loss_prot + r_loss_phos
            
            c_loss = classification_criterion(logits, batch_y.view_as(logits))
            
            val_loss += (total_r_loss + (alpha * c_loss)).item()
            val_recon += total_r_loss.item()
            val_class += c_loss.item()
            
    num_batches = len(test_loader)
    return val_loss / num_batches, val_recon / num_batches, val_class / num_batches


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    alpha: float,
    noise_factor: float,
    epochs: int,
    early_stopping_patience: int = 10,
) -> nn.Module:
    """
    Orchestrates the multi-omics training process including optimization, scheduling, and early stopping.
    """
    print(f"Training MOSAE on {device}")
    model = model.to(device)
    
    classification_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ae_params = list(model.prot_ae.parameters()) + list(model.phos_ae.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = optim.Adam([
        {'params': ae_params, 'weight_decay': 1e-4},
        {'params': classifier_params, 'weight_decay': 0.1}
    ], lr=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        delay_epochs = 15   # pure reconstruction
        warmup_epochs = 10  # Take 10 epochs to smoothly turn the volume up to max
        base_alpha = 0.15    # The maximum classifier weight you want to reach

        if epoch < delay_epochs:
            current_alpha = 0.0
        else:
            # Calculates how far along we are in the warmup phase (0.0 to 1.0)
            progress = (epoch - delay_epochs) / warmup_epochs
            current_alpha = base_alpha * min(1.0, progress)

        # Unpack the 3-tuple from training
        avg_train_loss, train_recon, train_class = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            classification_criterion=classification_criterion,
            device=device,
            noise_factor=noise_factor,
            alpha=current_alpha
        )
        
        # Unpack the 3-tuple from validation
        avg_val_loss, val_recon, val_class = _validate_model(
            model=model,
            test_loader=test_loader,
            classification_criterion=classification_criterion,
            device=device,
            alpha=current_alpha
        )
        
        if epoch >= delay_epochs:
            scheduler.step(val_class)
                
            if val_class < best_val_loss:
                best_val_loss = val_class
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at Epoch {epoch+1}! Restoring best weights.")
                break
        else:
            # Continuously save the latest weights during the pure reconstruction 
            # phase so we always have a valid model state to fall back on.
            best_model_weights = copy.deepcopy(model.state_dict())
        # ---------------------------------------------------------
            
        if (epoch + 1) % 3 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} (Recon: {train_recon:.4f}, Class: {train_class:.4f}) | "
                f"Val Loss: {avg_val_loss:.4f} (Recon: {val_recon:.4f}, Class: {val_class:.4f})"
            )

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    print("Training complete and best model restored.")
    return model
