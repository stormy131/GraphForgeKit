from torch.nn import MSELoss, CrossEntropyLoss


LOSSES = {
    "regression"    : MSELoss(),
    "classification": CrossEntropyLoss(),
}
