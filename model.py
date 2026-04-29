import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model



class TransformerModel(nn.Module):
    def __init__(self, n_positions, n_embd, n_layer, n_head, n_classes):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            vocab_size=1,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_classes = n_classes
        self.max_subcarriers = max(1, n_positions // 2)
        self._read_in = nn.Linear(self.n_classes, n_embd)
        self._subcarrier_embed = nn.Embedding(self.max_subcarriers, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, self.n_classes)

    @staticmethod
    def _combine(ys_b,xs_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        _, _, dim_y = ys_b.shape
        if dim_y < dim:
            padding_size = dim - dim_y
            padding = torch.zeros(bsize, points, padding_size, device=ys_b.device, dtype=ys_b.dtype)
            ys_b = torch.cat((ys_b, padding), dim=-1)
        zs = torch.stack((ys_b,xs_b), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, ys_batch, xs_batch, inds=None, subcarrier_indices=None):
        if inds is not None:
            inds = torch.tensor(inds)
            if max(inds) >= xs_batch.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(ys_batch, xs_batch)
        zs = zs.to(torch.float32)
        embeds = self._read_in(zs)

        if subcarrier_indices is not None:
            if subcarrier_indices.dim() == 1:
                subcarrier_indices = subcarrier_indices.unsqueeze(0).expand(xs_batch.shape[0], -1)
            subcarrier_indices = subcarrier_indices.to(device=xs_batch.device, dtype=torch.long)
            if subcarrier_indices.max() >= self.max_subcarriers or subcarrier_indices.min() < 0:
                raise ValueError("subcarrier_indices exceed the model's configured range")
            pair_indices = torch.repeat_interleave(subcarrier_indices, repeats=2, dim=1)
            embeds = embeds + self._subcarrier_embed(pair_indices)

        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        bsize, points, dim = ys_batch.shape
        return prediction[:, ::2, :]
