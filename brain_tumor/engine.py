"""Training loop with learning-rate scheduling and early stopping, and evaluation."""

import copy

import torch
from tqdm.auto import tqdm


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                scheduler=None, num_epochs=20, patience=5):
    """Train and validate each epoch. Stops after `patience` epochs without a lower
    validation loss and restores the weights with the lowest validation loss.

    Returns (model, history, best_val_loss).
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for phase in ("train", "val"):
            training = phase == "train"
            model.train() if training else model.eval()
            loader = train_loader if training else val_loader
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(loader, desc=phase.capitalize(), leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(training):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if training:
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (outputs.argmax(1) == labels).sum().item()

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects / len(loader.dataset)
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            print(f"  {phase.capitalize():5s} loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

            if not training:
                if scheduler is not None:
                    scheduler.step(epoch_loss)  # lower the learning rate when validation loss stalls
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping: restoring the best weights.")
            break

    print(f"Training complete. Best validation loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history, best_loss


@torch.no_grad()
def predict(model, loader, device, criterion=None):
    """Predictions and labels for a whole loader (and the mean loss if a criterion is given)."""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        if criterion is not None:
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    mean_loss = total_loss / len(loader.dataset) if criterion is not None else None
    return all_preds, all_labels, mean_loss
