import math
import torch
import torch.nn as nn


class VisitCodeTower(nn.Module):
    """
    One token-sequence tower per visit.
    Works for diagnoses, medications, lab all-code tokens, and vital all-code tokens.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, ids, mask, vmask):
        B, V, L = ids.shape

        x = self.emb(ids).reshape(B * V, L, -1)
        cls = self.cls.expand(B * V, 1, -1)
        seq = torch.cat([cls, x], dim=1)

        key_pad = torch.cat(
            [torch.zeros(B * V, 1, device=ids.device), 1 - mask.reshape(B * V, L)],
            dim=1,
        ).bool()

        h = self.enc(seq, src_key_padding_mask=key_pad)[:, 0, :]
        h = h.reshape(B, V, -1)

        return h * vmask.unsqueeze(-1)


class CrossDomainAttentionFusion(nn.Module):
    """
    Cross-domain attention over diagnosis, medication, lab, and vital towers.

    For each visit, the four tower outputs become a length-4 sequence:
        [diagnosis, medication, lab, vital]

    Multi-head self-attention lets domains interact before producing one fused
    visit representation.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1, n_domains=4):
        super().__init__()
        self.n_domains = n_domains
        self.domain_emb = nn.Embedding(n_domains, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_diag, h_med, h_lab, h_vit, domain_present, vmask):
        B, V, D = h_diag.shape

        x = torch.stack([h_diag, h_med, h_lab, h_vit], dim=2)

        domain_ids = torch.arange(self.n_domains, device=x.device)
        domain_pos = self.domain_emb(domain_ids).view(1, 1, self.n_domains, D)
        x = x + domain_pos

        x = x.reshape(B * V, self.n_domains, D)

        present = domain_present.reshape(B * V, self.n_domains) > 0.5

        all_missing = ~present.any(dim=1)
        present[all_missing] = True

        key_padding_mask = ~present

        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        present_f = present.float().unsqueeze(-1)
        fused = (x * present_f).sum(dim=1) / present_f.sum(dim=1).clamp_min(1.0)

        fused = fused.reshape(B, V, D)

        return fused * vmask.unsqueeze(-1)


class FourTowersTimeAwareAllCodes(nn.Module):
    """
    Proposed model:

    diagnosis all-code tower
    medication all-code tower
    lab all-code tower
    vital all-code tower
        -> cross-domain attention within each visit
        -> HTM-anchored temporal encoding
        -> learned visit position embedding
        -> patient-level transformer
        -> therapy response classifier
    """

    name = "four_towers_timeaware_allcodes"

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_visit_layers=2,
        n_patient_layers=2,
        max_visits=64,
        dropout=0.1,
        n_classes=2,
        time_fn="log",
        days_are_years=True,
        time_mode=None,
    ):
        super().__init__()

        self.max_visits = max_visits

        if time_mode is not None:
            time_fn = "log" if time_mode == "linear" else time_mode

        self.time_fn = time_fn
        self.days_are_years = days_are_years

        self.diag_tower = VisitCodeTower(
            vocab_size, d_model, n_heads, n_visit_layers, dropout
        )
        self.med_tower = VisitCodeTower(
            vocab_size, d_model, n_heads, n_visit_layers, dropout
        )
        self.lab_tower = VisitCodeTower(
            vocab_size, d_model, n_heads, n_visit_layers, dropout
        )
        self.vit_tower = VisitCodeTower(
            vocab_size, d_model, n_heads, n_visit_layers, dropout
        )

        self.cross_domain_fusion = CrossDomainAttentionFusion(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            n_domains=4,
        )

        self.pat_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pat_cls, std=0.02)

        self.time_proj = nn.Linear(1, d_model)
        self.pat_pos = nn.Embedding(max_visits + 2, d_model)

        pat_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.pat_enc = nn.TransformerEncoder(pat_layer, num_layers=n_patient_layers)

        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def encode_time(self, days, vmask):
        raw_days = days * 365.0 if self.days_are_years else days

        valid_days = raw_days.masked_fill(vmask < 0.5, -1000000000.0)
        anchor_day = valid_days.max(dim=1, keepdim=True)[0].clamp_min(0.0)

        d = (anchor_day - raw_days).clamp_min(0.0)

        if self.time_fn == "log":
            t = torch.log1p(d) / math.log1p(365.0)
        elif self.time_fn == "exp_30":
            t = torch.exp(-d / 30.0)
        elif self.time_fn == "exp_90":
            t = torch.exp(-d / 90.0)
        elif self.time_fn == "exp_180":
            t = torch.exp(-d / 180.0)
        else:
            raise ValueError(
                "Unknown time_fn: %r. Choose log | exp_30 | exp_90 | exp_180"
                % self.time_fn
            )

        return t * vmask

    def forward(self, batch):
        vmask = batch["vmask"]
        days = batch["days"]

        h_diag = self.diag_tower(batch["diag_ids"], batch["diag_mask"], vmask)
        h_med = self.med_tower(batch["med_ids"], batch["med_mask"], vmask)
        h_lab = self.lab_tower(batch["lab_ids"], batch["lab_mask"], vmask)
        h_vit = self.vit_tower(batch["vit_ids"], batch["vit_mask"], vmask)

        diag_present = (batch["diag_mask"].sum(dim=-1) > 0).float()
        med_present = (batch["med_mask"].sum(dim=-1) > 0).float()
        lab_present = (batch["lab_mask"].sum(dim=-1) > 0).float()
        vit_present = (batch["vit_mask"].sum(dim=-1) > 0).float()

        domain_present = torch.stack(
            [diag_present, med_present, lab_present, vit_present],
            dim=-1,
        )

        h_visit = self.cross_domain_fusion(
            h_diag,
            h_med,
            h_lab,
            h_vit,
            domain_present,
            vmask,
        )

        days_enc = self.encode_time(days, vmask)
        tpe = self.time_proj(days_enc.unsqueeze(-1))

        B, V, _ = h_visit.shape

        pos = torch.arange(V, device=h_visit.device).unsqueeze(0).expand(B, -1)
        pos = pos.clamp_max(self.pat_pos.num_embeddings - 1)

        pat_in = h_visit + tpe + self.pat_pos(pos)

        pat_cls = self.pat_cls.expand(B, 1, -1)
        pat_seq = torch.cat([pat_cls, pat_in], dim=1)

        pat_mask = torch.cat(
            [torch.ones(B, 1, device=h_visit.device), vmask],
            dim=1,
        )

        h_pat = self.pat_enc(pat_seq, src_key_padding_mask=pat_mask < 0.5)

        return self.cls(self.dropout(h_pat[:, 0, :]))
