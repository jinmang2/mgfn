import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record

from tqdm import tqdm
from mgfn import mgfn
from dataset import Dataset

from train import train
from test import test


if __name__ == "__main__":
    batch_size = 16
    max_epochs = 1000

    train_nloader = DataLoader(
        Dataset(test_mode=False, is_normal=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    train_aloader = DataLoader(
        Dataset(test_mode=False, is_normal=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    test_loader = DataLoader(
        Dataset(test_mode=True),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = mgfn()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if not os.path.exists("./ckpt"):
        os.makedirs("./ckpt")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    test_info = {"epoch": [], "test_AUC": [], "test_PR": []}

    best_AUC = -1
    best_PR = -1  # put your own path here

    iterator = 0
    for step in tqdm(
        range(1, max_epochs + 1), total=max_epochs, dynamic_ncols=True
    ):
        cost, loss_smooth, loss_sparse = train(
            train_nloader,
            train_aloader,
            model,
            batch_size,
            optimizer,
            device,
            iterator,
        )

        if step % 1 == 0 and step > 0:
            auc, pr_auc = test(test_loader, model, device)

            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            test_info["test_PR"].append(pr_auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(
                    model.state_dict(),
                    "/mgfn{}-i3d.pkl".format(step),
                )
                save_best_record(
                    test_info,
                    os.path.join("/{}-step-AUC.txt".format(step)),
                )
    torch.save(model.state_dict(), "./ckpt/mgfn_final.pkl")
