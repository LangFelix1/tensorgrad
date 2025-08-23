from tensor import no_grad

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