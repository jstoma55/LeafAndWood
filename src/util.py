def accuracy(predicted, labels):
    p = predicted.detach().cpu().argmax(dim=1).numpy()
    l = labels.detach().numpy()
    correct = (p == l).sum()
    total = l.shape[0]
    return correct, total
