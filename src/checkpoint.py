import pickle

def save_checkpoint(path, model, optimizer=None, epoch=None):
    checkpoint = {
        "model": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path, model, optimizer=None):
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint.get("epoch", None)