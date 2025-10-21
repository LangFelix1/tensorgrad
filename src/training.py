from src.tensor import no_grad
from src.lr_scheduler import ReduceLROnPlateau

def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0.
    total_samples = 0

    for x, y in dataloader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item() * x.shape[0]
        total_samples += x.shape[0]

    return total_loss / total_samples

def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.
    total_samples = 0

    with no_grad():
        for x, y in dataloader:
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.data.item() * x.shape[0]
            total_samples += x.shape[0]

    return total_loss / total_samples

def fit(model, train_loader, loss_fn, optimizer, num_epochs=10, val_loader=None, scheduler=None):
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, loss_fn)
        else:
            val_loss = None

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss if val_loss is not None else train_loss)
            else:
                scheduler.step()

        lr_str = f" lr={optimizer.lr:.6g}"
        print(f"Epoch {epoch+1}/{num_epochs},{lr_str} Train: {train_loss:.6f}"
              + (f"  Val: {val_loss:.6f}" if val_loss is not None else ""))