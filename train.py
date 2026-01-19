import random
import time
import numpy as np
import pandas as pd
from src.model import POM, SPMM, EndToEndModel
from src.early_stop import EarlyStopping
from src.tools import get_auroc_ap, filtered_score, format_duration
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import KLDivLoss
import matplotlib.pyplot as plt
import os, sys
import argparse


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MixtureDataset(Dataset):
    def __init__(self, df):
        self.smiles = df["smiles"].tolist()
        self.labels = df.iloc[:, 1:].values.astype("float32")
        self.is_molecule = [(";" not in smi) for smi in self.smiles]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {
            "smiles": self.smiles[idx],
            "label": torch.tensor(self.labels[idx]),
            "is_molecule": self.is_molecule[idx]
        }

class MLD(nn.Module):
    def __init__(self, reduction="batchmean", eps=1e-8):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.criterion = KLDivLoss(reduction="none")

    def forward(self, student, teacher, N):
        student = torch.clamp(student, min=self.eps, max=1-self.eps)
        teacher = torch.clamp(teacher, min=self.eps, max=1-self.eps)
        loss = self.criterion(torch.log(student), teacher) + self.criterion(torch.log(1 - student), 1 - teacher)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "batchmean":
            loss = loss.sum() / N
        elif self.reduction == "mean":
            loss = loss.mean()
        else:
            raise AttributeError
        return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)

    args = parser.parse_args()

    phase = args.phase

    if phase == "aromma":
        data_name = "mixture"
    elif phase == "aromma_p78":
        data_name = "mixture_p78"
    elif phase == "aromma_p152":
        data_name = "mixture_p152"

    save_dir = phase
    os.makedirs(f"results/{save_dir}", exist_ok=True)

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/gslf_138.txt", "r") as f:
        labels_gslf = f.read().splitlines()
    with open("data/bp_74.txt", "r") as f:
        labels_bp = f.read().splitlines()    
    with open("data/labels_152.txt", "r") as f:
        labels_mixture = f.read().splitlines()

    gslf_indices = [labels_mixture.index(lbl) for lbl in labels_gslf]
    bp_indices = [labels_mixture.index(lbl) for lbl in labels_bp]

    batch_size = 128
    num_epoch = 1000
    num_labels = 152
    sattn_hidden_layer = 196
    cattn_hidden_dim = 384
    learning_rate = 4e-5

    auroc_folds, ap_folds = [], []
    gslf_auroc_folds, gslf_ap_folds = [], []
    bp_auroc_folds, bp_ap_folds = [], []

    train_loss_folds_f, valid_loss_folds_f = [], []
    total_auroc_folds_f, gslf_auroc_folds_f, bp_auroc_folds_f = [], [], []

    start_time = time.time()
    for i in range(1, 6):
        df_train = pd.read_csv(f"data/{data_name}/fold{i}/train.csv")
        df_valid = pd.read_csv(f"data/{data_name}/fold{i}/valid.csv")
        df_test = pd.read_csv(f"data/mixture/fold{i}/test.csv")

        train_loader = DataLoader(MixtureDataset(df_train), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(MixtureDataset(df_valid), batch_size=batch_size)
        test_loader = DataLoader(MixtureDataset(df_test), batch_size=batch_size)
        
        model_t = POM()
        embedder = SPMM(r=4, lora_alpha=8)
        model_s = EndToEndModel(embedder=embedder, sattn_hidden_dim=sattn_hidden_layer, cattn_hidden_dim=cattn_hidden_dim, num_heads=4, num_labels=num_labels).to(device)
        criterion = nn.BCEWithLogitsLoss()
        kl_loss = MLD()
        optimizer = torch.optim.Adam(model_s.parameters(), lr=learning_rate)
        es = EarlyStopping(model_s, patience=20, mode="minimize")

        total = sum(p.numel() for p in model_s.parameters())
        trainable = sum(p.numel() for p in model_s.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}") 
        print(f"Trainable parameters: {trainable:,}")

        total_auroc_arr, gslf_auroc_arr, bp_auroc_arr = [], [], []
        train_loss_arr, valid_loss_arr = [], []

        for epoch in range(1, num_epoch+1):
            model_s.train()
            train_loss = 0
            for train_batch in train_loader:
                smiles, label, is_molecule = train_batch["smiles"], train_batch["label"].to(device), train_batch["is_molecule"]
                logit_s = model_s(smiles)

                N = label.shape[0]
                
                molecule_indices = is_molecule.nonzero(as_tuple=True)[0]

                logits_mol = logit_s[is_molecule]
                labels_mol = label[is_molecule]

                logits_mix = logit_s[~is_molecule]
                labels_mix = label[~is_molecule]

                mix_loss = criterion(logits_mix, labels_mix)

                if molecule_indices.numel() == 0:
                    total_loss = mix_loss
                else:
                    # molecule
                    mol_smi_arr = [
                        smi
                        for smi, is_mol in zip(smiles, is_molecule.tolist())
                        if is_mol
                    ]

                    model_probs = torch.sigmoid(logits_mol)[:, gslf_indices]
                    gt_labels = label[molecule_indices][:, gslf_indices]
                    
                    with torch.no_grad():
                        if isinstance(model_t, POM):
                            logit_t = model_t.forward(mol_smi_arr)
                        else:
                            logit_t = model_t(mol_smi_arr)[:, gslf_indices]

                    prob_t = torch.sigmoid(logit_t)

                    kd_loss = kl_loss.forward(model_probs, prob_t, N)
                    mol_loss = criterion(logits_mol, labels_mol) + kd_loss 

                    total_loss = 0.5 * mix_loss + 0.5 * mol_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            train_loss /= len(train_loader)
            train_loss_arr.append(train_loss)

            model_s.eval()
            valid_loss = 0
            valid_pred = []
            valid_true = []
            with torch.no_grad():
                for valid_batch in valid_loader:
                    smiles, label = valid_batch["smiles"], valid_batch["label"].to(device)
                    logit = model_s(smiles)
                    loss = criterion(logit, label)
                    valid_loss += loss.item()

                    prob = torch.sigmoid(logit)
                    valid_pred.append(prob.cpu())
                    valid_true.append(label.cpu())
            valid_loss /= len(valid_loader)
            valid_loss_arr.append(valid_loss)

            valid_pred = torch.cat(valid_pred).numpy()
            valid_true = torch.cat(valid_true).numpy()

            valid_smiles = sum([batch["smiles"] for batch in valid_loader], [])
            gslf_mask = [(";" not in smi) for smi in valid_smiles]
            bp_mask = [(";" in smi) for smi in valid_smiles]
            
            total_auc, total_ap = get_auroc_ap(valid_true, valid_pred)
            gslf_auc, gslf_ap = filtered_score(valid_true, valid_pred, gslf_mask, gslf_indices)
            bp_auc, bp_ap = filtered_score(valid_true, valid_pred, bp_mask, bp_indices)

            total_auroc_arr.append(total_auc)
            gslf_auroc_arr.append(gslf_auc)
            bp_auroc_arr.append(bp_auc)

            print(f"Fold{i} | Epoch {epoch:03d} | [TRAIN] loss={train_loss:.4f} | [VALID] loss={valid_loss:.4f}, (total) AUROC={total_auc:.4f}, AP={total_ap:.4f} (gs-lf) AUROC={gslf_auc:.4f}, AP={gslf_ap:.4f} (bp) AUROC={bp_auc:.4f}, AP={bp_ap:.4f}")

            if es.check_criteria(valid_loss, model_s):
                print(f"Early stop reached at epoch {epoch} with loss {valid_loss:.4f}")
                break

        train_loss_folds_f.append(train_loss_arr)
        valid_loss_folds_f.append(valid_loss_arr)

        total_auroc_folds_f.append(total_auroc_arr)
        gslf_auroc_folds_f.append(gslf_auroc_arr)
        bp_auroc_folds_f.append(bp_auroc_arr)

        best_model_dict, best_model_value = es.restore_best()
        model_s.load_state_dict(best_model_dict)
        torch.save(model_s.state_dict(), f"results/{save_dir}/checkpoint_fold{i}_{best_model_value:.4f}.pt")

        model_s.eval()
        test_pred = []
        test_true = []
        with torch.no_grad():
            for test_batch in test_loader:
                smiles, label = test_batch["smiles"], test_batch["label"].to(device)
                logit = model_s(smiles)
                prob = torch.sigmoid(logit)
                test_pred.append(prob.cpu())
                test_true.append(label.cpu())
        test_pred = torch.cat(test_pred).numpy()
        test_true = torch.cat(test_true).numpy()

        test_smiles = sum([batch["smiles"] for batch in test_loader], [])
        gslf_mask = [(";" not in smi) for smi in test_smiles]
        bp_mask = [(";" in smi) for smi in test_smiles]
        
        total_auc, total_ap = get_auroc_ap(test_true, test_pred)
        gslf_auc, gslf_ap = filtered_score(test_true, test_pred, gslf_mask, gslf_indices)
        bp_auc, bp_ap = filtered_score(test_true, test_pred, bp_mask, bp_indices)

        auroc_folds.append(total_auc)
        ap_folds.append(total_ap)
        gslf_auroc_folds.append(gslf_auc)
        gslf_ap_folds.append(gslf_ap)
        bp_auroc_folds.append(bp_auc)
        bp_ap_folds.append(bp_ap)

        print(
            f"=> [Total] AUROC={total_auc:.4f}, AP={total_ap:.4f} [GS-LF] AUROC={gslf_auc:.4f}, AP={gslf_ap:.4f} [BP] AUROC={bp_auc:.4f}, AP={bp_ap:.4f}"
        )
    end_time = time.time()

    original_stdout = sys.stdout
    with open(f"results/{save_dir}/summary.txt", 'w') as f:
        sys.stdout = f 
        print(f"Total parameters: {total:,}") 
        print(f"Trainable parameters: {trainable:,}")
        
        print(f"TIME TAKEN={format_duration(end_time-start_time)}s")
        print("Fold Summary")
        print(f"    o AUROC: {np.mean(auroc_folds):.4f} ± {np.std(auroc_folds):.4f}")
        print(f"    o AP: {np.mean(ap_folds):.4f} ± {np.std(ap_folds):.4f}")
        print(f"GS-LF) AUROC: {np.mean(gslf_auroc_folds):.4f} ± {np.std(gslf_auroc_folds):.4f} AP: {np.mean(gslf_ap_folds):.4f} ± {np.std(gslf_ap_folds):.4f}")
        print(f"BP) AUROC: {np.mean(bp_auroc_folds):.4f} ± {np.std(bp_auroc_folds):.4f} AP: {np.mean(bp_ap_folds):.4f} ± {np.std(bp_ap_folds):.4f}")

    sys.stdout = original_stdout

    fig, axes = plt.subplots(1, 5, figsize=(5 * 5, 5), sharey=True)

    for i in range(5):
        ax = axes[i]
        ax.plot(train_loss_folds_f[i], label="Train")
        ax.plot(valid_loss_folds_f[i], label="Valid")
        ax.set_xlabel("Epoch")
        ax.set_title(f"Fold {i+1}")
        ax.legend()
        ax.grid()

    axes[0].set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"results/{save_dir}/loss.png")
    plt.close()

    fig, axes = plt.subplots(1, 5, figsize=(5 * 5, 5), sharey=True)

    for i in range(5):
        ax = axes[i]
        ax.plot(total_auroc_folds_f[i], label="Total")
        ax.plot(gslf_auroc_folds_f[i], label="GS-LF")
        ax.plot(bp_auroc_folds_f[i], label="Blend-Pair")
        ax.set_xlabel("Epoch")
        ax.set_title(f"Fold {i+1}")
        ax.legend()
        ax.grid()

    axes[0].set_ylabel("AUROC")
    plt.tight_layout()
    plt.savefig(f"results/{save_dir}/auroc.png")
    plt.close()
