import pickle

def save_checkpoint(path, model, optimizer=None, epoch=None, scheduler=None):
    checkpoint = {
        "model": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint.get("epoch", None)