from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
import numpy as np

from mgfn import mgfn
from dataset import Dataset


def test(dataloader, model, args, device):
    plt.clf()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        featurelen = []
        for i, inputs in tqdm(enumerate(dataloader)):

            input = inputs[0].to(device)
            input = input.permute(0, 2, 1, 3)
            _, _, _, _, logits = model(input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            featurelen.append(len(sig))
            pred = torch.cat((pred, sig))

        gt = np.load(args.gt)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        fpr, tpr, _ = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        print("pr_auc : " + str(pr_auc))
        print("rec_auc : " + str(rec_auc))
        return rec_auc, pr_auc


if __name__ == "__main__":
    model = mgfn()
    test_loader = DataLoader(
        Dataset(test_mode=True),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device
    model_dict = model.load_state_dict(
        {k.replace("module.", ""): v for k, v in torch.load("mgfn_ucf.pkl").items()}
    )
    auc = test(test_loader, model, device)
