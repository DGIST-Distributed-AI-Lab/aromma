import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.dropout = nn.Dropout(0.4)

    def forward(self, emb, mask):
        emb = self.proj(emb)
        emb = self.relu(emb)

        attn_out, _ = self.attn(emb, emb, emb, key_padding_mask=~mask)
        return self.dropout(attn_out)


class CrossAttentionAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.query)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, emb, mask):
        emb = self.proj(emb)
        emb = self.relu(emb)
        B = emb.size(0)
        q = self.query.expand(B, -1, -1)  # learnable query

        out, _ = self.attn(q, emb, emb, key_padding_mask=~mask)
        return self.norm(out.squeeze(1))


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.mlp = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        return self.mlp(x)


class EndToEndModel(nn.Module):
    def __init__(
        self, embedder, sattn_hidden_dim, cattn_hidden_dim, num_heads, num_labels
    ):
        super().__init__()
        self.embedder = embedder
        self.output_dim = embedder.output_dim

        self.self_attn = SelfAttention(
            input_dim=self.output_dim, hidden_dim=sattn_hidden_dim, num_heads=num_heads
        )
        self.cross_attn = CrossAttentionAggregator(
            input_dim=sattn_hidden_dim, hidden_dim=cattn_hidden_dim, num_heads=num_heads
        )
        self.classifier = MLPClassifier(
            input_dim=cattn_hidden_dim, num_labels=num_labels
        )

    def forward(self, smiles_batch, return_emb=False):
        smiles_flat, split_lens = [], []
        for smi in smiles_batch:
            if ";" in smi:
                parts = smi.split(";")
            else:
                parts = [smi]
            smiles_flat.extend(parts)
            split_lens.append(len(parts))

        all_embs = self.embedder.forward(smiles_flat)

        embeddings, masks = [], []
        idx = 0
        for n_parts in split_lens:
            part_embs = all_embs[idx : idx + n_parts]
            part_mask = [True] * n_parts

            if n_parts < 2:
                pad = torch.zeros((1, part_embs.size(1)), device=self.embedder.device)
                part_embs = torch.cat([part_embs, pad], dim=0)
                part_mask += [False]
            embeddings.append(part_embs)
            masks.append(torch.tensor(part_mask, device=self.embedder.device))
            idx += n_parts

        emb_tensor = torch.stack(embeddings)
        mask_tensor = torch.stack(masks)

        self_attn_out = self.self_attn(emb_tensor, mask_tensor)
        cross_attn_out = self.cross_attn(self_attn_out, mask_tensor)
        logits = self.classifier(cross_attn_out)

        if return_emb:
            return cross_attn_out

        return logits
