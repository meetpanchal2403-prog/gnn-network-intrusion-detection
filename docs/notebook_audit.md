# Notebook Audit

- Notebook: `analysis\full_capstone.ipynb`
- Total cells: 170

## Code cell inventory

### Cell 78
- Matching terms: GraphSAGEEncoder, GraphMAE

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader

HIDDEN_DIM = 128
EMBED_DIM = 128
MASK_RATE = 0.5
GAMMA = 2.0

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h

class GraphMAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        self.remask_token = nn.Parameter(torch.zeros(embed_dim))
        self.encoder = GraphSAGEEncoder(in_dim, hidden_dim, embed_dim)
        self.decoder = SAGEConv(embed_dim, in_dim)

    def forward(self, x, edge_index, active, mask_rate=MASK_RATE):
        active_idx = active.nonzero(as_tuple=True)[0]
        n_mask = max(1, int(len(active_idx) * mask_rate))
        perm = active_idx[torch.randperm(len(active_idx), device=x.device)]
        mask_idx = perm[:n_mask]

        x_corrupt = x.clone()
        x_corrupt[mask_idx] = self.mask_token

        h = self.encoder(x_corrupt, edge_index)
        h_remask = h.clone()
        h_remask[mask_idx] = self.remask_token
        recon = self.decoder(h_remask, edge_index)

        target = x[mask_idx]
        pred = recon[mask_idx]
        cos = F.cosine_similarity(pred, target, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
        loss = ((1 - cos) ** GAMMA).mean()
        return loss, h

train_loader = DataLoader(all_graphs['train'], batch_size=32, shuffle=True)

model = GraphMAE(FEATURE_DIM, HIDDEN_DIM, EMBED_DIM).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

N_EPOCHS = 30
history = []
for epoch in range(N_EPOCHS):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        if batch.active.sum() < 2:
            continue
        opt.zero_grad()
        loss, _ = model(batch.x, batch.edge_index, batch.active)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    avg = total_loss / max(n_batches, 1)
    history.append(avg)
    if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
        print(f'epoch {epoch:3d}  recon loss {avg:.4f}')

print('\nLoss went from', round(history[0], 4),
```

### Cell 79
- Matching terms: pretrain

```python
torch.save({
    'encoder_state_dict': model.encoder.state_dict(),
    'in_dim': FEATURE_DIM, 'hidden_dim': HIDDEN_DIM, 'embed_dim': EMBED_DIM,
    'blocks': ('flow', 'log', 'emb'),
}, os.path.join(OUT_DIR, 'pretrained_encoder.pt'))
print('Saved pretrained encoder to', os.path.join(OUT_DIR, 'pretrained_encoder.pt'))
```

### Cell 84
- Matching terms: GraphSAGEEncoder, GraphMAE, EdgeNIDSClassifier, pretrain

```python
class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels, edge_dim, out_channels):
        super().__init__(aggr='mean')
        self.lin_msg = nn.Linear(in_channels + edge_dim, out_channels)
        self.lin_root = nn.Linear(in_channels, out_channels)
    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out + self.lin_root(x)
    def message(self, x_j, edge_attr):
        return F.relu(self.lin_msg(torch.cat([x_j, edge_attr], dim=-1)))

class EdgeSAGEEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = EdgeSAGEConv(in_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeSAGEConv(hidden_dim, edge_dim, out_dim)
    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        return self.conv2(h, edge_index, edge_attr)

class EdgeGraphMAE(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, embed_dim):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        self.remask_token = nn.Parameter(torch.zeros(embed_dim))
        self.encoder = EdgeSAGEEncoder(in_dim, edge_dim, hidden_dim, embed_dim)
        self.decoder = EdgeSAGEConv(embed_dim, edge_dim, in_dim)

    def forward(self, x, edge_index, edge_attr, active, mask_rate=0.5, gamma=2.0):
        active_idx = active.nonzero(as_tuple=True)[0]
        if len(active_idx) < 2:
            return None
        n_mask = max(1, int(len(active_idx) * mask_rate))
        perm = active_idx[torch.randperm(len(active_idx), device=x.device)]
        mask_idx = perm[:n_mask]
        x_corrupt = x.clone(); x_corrupt[mask_idx] = self.mask_token
        h = self.encoder(x_corrupt, edge_index, edge_attr)
        h_remask = h.clone(); h_remask[mask_idx] = self.remask_token
        recon = self.decoder(h_remask, edge_index, edge_attr)
        target, pred = x[mask_idx], recon[mask_idx]
        cos = F.cosine_similarity(pred, target, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
        return ((1 - cos) ** gamma).mean()

class EdgeNIDSClassifier(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, embed_dim, pretrained_state=None):
        super().__init__()
        self.encoder = EdgeSAGEEncoder(in_dim, edge_dim, hidden_dim, embed_dim)
        if pretrained_state is not None:
            self.encoder.load_state_dict(pretrained_state)
        self.head = nn.Seq
```

### Cell 85
- Matching terms: train_model, evaluate_logits

```python
def pos_weight_for(graphs):
    y_all = torch.cat([g.y[g.active] for g in graphs])
    pos = y_all.sum().item(); neg = len(y_all) - pos
    return torch.tensor([neg / max(pos, 1)], device=DEVICE)

def evaluate_logits(model, graphs):
    model.eval()
    all_logits, all_y, all_active, all_scenario, all_window, all_block = [], [], [], [], [], []
    with torch.no_grad():
        for g in graphs:
            g = g.to(DEVICE)
            logits = model(g.x, g.edge_index, g.edge_attr)
            all_logits.append(logits.cpu()); all_y.append(g.y.cpu()); all_active.append(g.active.cpu())
            all_scenario += [g.scenario] * g.x.shape[0]
            all_window += [g.window_id] * g.x.shape[0]
            all_block += [g.block_id] * g.x.shape[0]
    return (torch.cat(all_logits), torch.cat(all_y), torch.cat(all_active),
            np.array(all_scenario), np.array(all_window), np.array(all_block))

def _period_of(sc_window_label_table, window_id):
    pass  # not used directly -- periods are read from graph construction's window_periods via scenario dicts below

def macro_f1_for_selection(model, graphs, fold):
    # exfil+chain macro F1 on val, used for early-stop checkpoint selection (clean excluded, per the
    # earlier fix: pooled/undifferentiated selection silently ignores the rare, important classes)
    logits, y, active, scenario, window, _ = evaluate_logits(model, graphs)
    pred = (logits > 0).float()
    f1s = []
    for period in ['exfil', 'chain']:
        mask = active.numpy().astype(bool).copy()
        period_mask = np.zeros(len(mask), dtype=bool)
        for sc in set(scenario):
            sc_mask = scenario == sc
            wp = fold['scenarios'][sc]['window_periods']
            wid_to_period = {int(w): wp[i] for i, w in enumerate(fold['scenarios'][sc]['window_ids'])}
            this_period = np.array([wid_to_period.get(int(w), 'clean') == period for w in window[sc_mask]])
            period_mask[sc_mask] = this_period
        m = mask & period_mask
        if m.sum() == 0: continue
        yt, yp = y.numpy()[m], pred.numpy()[m]
        if yt.sum() == 0 and yp.sum() == 0: f1 = 1.0
        else: _, _, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0

def train_model(model, train_graphs, val_graphs, fold, n_epochs=40, lr=1e-3, patience=8, tag=''):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

```

### Cell 86
- Matching terms: evaluate_logits

```python
def _metric(yt, yp):
    if yt.sum() == 0 and yp.sum() == 0:
        return 1.0, 1.0, 1.0
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
    return p, r, f1

def block_bootstrap_ci(y_true, y_pred, block_ids, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(block_ids)
    block_to_idx = {b: np.flatnonzero(block_ids == b) for b in unique_blocks}
    f1s = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_to_idx[b] for b in sampled])
        _, _, f1 = _metric(y_true[idx], y_pred[idx])
        f1s.append(f1)
    return float(np.mean(f1s)), float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))

def period_report(model, test_graphs, fold, test_scenario, n_boot=1000):
    logits, y, active, scenario, window, block = evaluate_logits(model, test_graphs)
    pred = (logits > 0).float().numpy()
    yt_all = y.numpy(); act = active.numpy().astype(bool)
    wp = fold['scenarios'][test_scenario]['window_periods']
    wid_to_period = {int(w): wp[i] for i, w in enumerate(fold['scenarios'][test_scenario]['window_ids'])}
    periods = np.array([wid_to_period.get(int(w), 'clean') for w in window])

    rows = []
    for period in ['exfil', 'chain', 'clean']:
        m = act & (periods == period)
        if m.sum() == 0: continue
        yt, yp, blk = yt_all[m], pred[m], block[m]
        p, r, f1 = _metric(yt, yp)
        f1_mean, f1_lo, f1_hi = block_bootstrap_ci(yt, yp, blk, n_boot=n_boot)
        n_pos = int(yt.sum())
        rows.append(dict(test_scenario=test_scenario, period=period, n=int(m.sum()), n_pos=n_pos,
                          precision=round(p,3), recall=round(r,3), f1=round(f1,3),
                          f1_boot_mean=round(f1_mean,3), f1_ci_lo=round(f1_lo,3), f1_ci_hi=round(f1_hi,3)))
    return pd.DataFrame(rows)
```

### Cell 87
- Matching terms: GraphMAE, EdgeNIDSClassifier, train_model, pretrain

```python
N_SEEDS = 3          # pilot with 3 first; raise to 5 once you've seen how long this takes
SEEDS = [0, 1, 2, 3, 4][:N_SEEDS]
MODEL_ARCH = 'edge'
KEEP_BLOCKS = ('flow', 'log', 'emb')
USE_PRETRAINED = False   # the ablation: False = train the encoder from scratch, no GraphMAE pretraining
MODE_NAME = 'multimodal' if len(KEEP_BLOCKS) == 3 else 'flow_only'
MODEL_TAG = f'{MODE_NAME}_{MODEL_ARCH}' + ('' if USE_PRETRAINED else '_scratch')
RESULTS_PATH = os.path.join(OUT_DIR, f'loso_results_{MODEL_TAG}.csv')

MAEClass = EdgeGraphMAE if MODEL_ARCH == 'edge' else GraphMAE
ClassifierClass = EdgeNIDSClassifier if MODEL_ARCH == 'edge' else NIDSClassifier

# resume support: skip (fold, seed) pairs already completed if this gets interrupted and rerun
if os.path.exists(RESULTS_PATH):
    done = pd.read_csv(RESULTS_PATH)
    done_pairs = set(zip(done['test_scenario'], done['seed']))
    print(f'Resuming: {len(done_pairs)} (fold, seed) pairs already completed.')
else:
    done = pd.DataFrame()
    done_pairs = set()

all_results = [done] if len(done) else []

for test_scenario in sorted(SCENARIOS):
    fold_path = os.path.join(OUT_DIR, f'loso_fold_test-{test_scenario}_w{WINDOW_MINUTES}m.pkl')
    with open(fold_path, 'rb') as f:
        fold = pickle.load(f)

    train_graphs = build_graphs_from_fold(fold, 'train', keep_blocks=KEEP_BLOCKS)
    val_graphs = build_graphs_from_fold(fold, 'val', keep_blocks=KEEP_BLOCKS)
    test_graphs = build_graphs_from_fold(fold, 'test', keep_blocks=KEEP_BLOCKS)
    in_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]

    # pretrain ONCE per fold (not per seed) -- reused across this fold's seeds below
    print(f'\n=== Fold test={test_scenario}: pretraining encoder ===')
    set_seed(0)
    if USE_PRETRAINED:
        pretrained_state = pretrain_encoder(MAEClass, train_graphs, in_dim, edge_dim, tag=test_scenario)
    else:
        pretrained_state = None
        print(f'  (skipping GraphMAE pretraining -- from-scratch ablation)')

    for seed in SEEDS:
        if (test_scenario, seed) in done_pairs:
            print(f'  seed {seed}: already done, skipping')
            continue
        print(f'\n--- test={test_scenario}, seed={seed} ---')
        set_seed(seed)
        model = (EdgeNIDSClassifier(in_dim, edge_dim, 128, 128, pretrained_state=pretrained_state)
                 if MODEL_ARCH == 'edge' else
                 NIDSClassifier(in_dim, 128, 128, pretrained
```

### Cell 90
- Matching terms: GraphMAE, EdgeNIDSClassifier, train_model, pretrain

```python
def evaluate_detailed(model, test_graphs, fold, test_scenario):
    model.eval()
    hosts = fold['scenarios'][test_scenario]['hosts']
    label_table = fold['scenarios'][test_scenario]['label_table']
    # index label_table for fast lookup: (host, window_id) -> labels list
    label_lookup = {(r.host, int(r.window_id)): r.labels for r in label_table.itertuples()}

    rows = []
    with torch.no_grad():
        for g in test_graphs:
            g_dev = g.to(DEVICE)
            logits = model(g_dev.x, g_dev.edge_index, g_dev.edge_attr).cpu().numpy()
            y = g_dev.y.cpu().numpy(); active = g_dev.active.cpu().numpy().astype(bool)
            pred = (logits > 0).astype(int)
            for node_idx in np.flatnonzero(active):
                host = hosts[node_idx]
                labs = label_lookup.get((host, g.window_id), [])
                rows.append(dict(
                    test_scenario=test_scenario, window_id=g.window_id, host=host,
                    host_role=canonical_host_role(host),
                    y_true=int(y[node_idx]), y_pred=int(pred[node_idx]),
                    labels=labs,
                ))
    return pd.DataFrame(rows)

DETAIL_SEED = 0
all_detail_rows = []

for test_scenario in sorted(SCENARIOS):
    fold_path = os.path.join(OUT_DIR, f'loso_fold_test-{test_scenario}_w{WINDOW_MINUTES}m.pkl')
    with open(fold_path, 'rb') as f:
        fold = pickle.load(f)

    train_graphs = build_graphs_from_fold(fold, 'train', keep_blocks=('flow', 'log', 'emb'))
    val_graphs = build_graphs_from_fold(fold, 'val', keep_blocks=('flow', 'log', 'emb'))
    test_graphs = build_graphs_from_fold(fold, 'test', keep_blocks=('flow', 'log', 'emb'))
    in_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]

    print(f'\n=== {test_scenario}: pretrain + train (seed={DETAIL_SEED}) ===')
    set_seed(0)
    pretrained_state = pretrain_encoder(EdgeGraphMAE, train_graphs, in_dim, edge_dim, tag=test_scenario)
    set_seed(DETAIL_SEED)
    model = EdgeNIDSClassifier(in_dim, edge_dim, 128, 128, pretrained_state=pretrained_state)
    model, val_f1 = train_model(model, train_graphs, val_graphs, fold, tag=test_scenario)
    print(f'  val macro-F1: {val_f1:.4f}')

    detail = evaluate_detailed(model, test_graphs, fold, test_scenario)
    all_detail_rows.append(detail)
    pd.concat(all_detail_rows, ignore_index=True).to_csv(os.path.join(OUT_DIR, 'loso_detail_predictions.csv'), index=False)
    print(f'  {len(detail)} n
```

### Cell 94
- Matching terms: pretrain

```python
import os, pickle, collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_recall_fscore_support

OUT_DIR = r"G:\My Drive\nids-capstone\data\ait\processed"
WINDOW_MINUTES = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

with open(os.path.join(OUT_DIR, f'multi_scenario_graph_dataset_w{WINDOW_MINUTES}m.pkl'), 'rb') as f:
    dataset = pickle.load(f)

ckpt = torch.load(os.path.join(OUT_DIR, 'pretrained_encoder.pt'), map_location=DEVICE, weights_only=False)
HIDDEN_DIM, EMBED_DIM, FEATURE_DIM = ckpt['hidden_dim'], ckpt['embed_dim'], ckpt['in_dim']
print('Pretrained encoder:', FEATURE_DIM, '->', HIDDEN_DIM, '->', EMBED_DIM)

EXFIL_LABELS = {'data exfiltration', 'attacker', 'dnsteal', 'dnsteal-received'}
CHAIN_LABELS = {
    'service_scan', 'dirb_scan', 'wpscan', 'dns_brute_force_start', 'host_discover_dmz',
    'host_discover_local', 'online_cracking', 'upload_rce_shell', 'dump_wp_users',
    'check_whoami', 'check_uname_r', 'read_profile', 'check_who', 'check_meminfo', 'check_uname_a',
    'check_user_id', 'check_df', 'check_netstat_nat', 'check_id', 'read_resolv', 'check_netstat_t',
    'list_home', 'check_last', 'list_web_dir', 'check_date', 'list_www', 'check_netstat_l', 'clear',
    'check_wp_config', 'check_ps_a', 'read_passwd', 'check_release', 'check_cpuinfo', 'read_group',
    'check_uptime', 'check_network_config', 'update/command on unassigned port', 'check_pwd', 'list_l',
}

def window_periods(scen):
    lt = scen['label_table']
    has_chain = lt['labels'].apply(lambda labs: any(l in CHAIN_LABELS for l in labs))
    has_exfil = lt['labels'].apply(lambda labs: any(l in EXFIL_LABELS for l in labs))
    chain_windows = set(lt.loc[has_chain, 'window_id'].astype(int))
    exfil_windows = set(lt.loc[has_exfil, 'window_id'].astype(int)) - chain_windows
    periods = {}
    for w in scen['window_ids']:
        w = int(w)
        periods[w] = 'chain' if w in chain_windows else ('exfil' if w in exfil_windows else 'clean')
    return periods

scenario_periods = {sc: window_periods(scen) for sc, scen in dataset['scenarios'].items()}
for sc, p in scenario_periods.items():
    print(sc, dict(collections.Counter(p.values())))
```

### Cell 96
- Matching terms: GraphSAGEEncoder, train_model, evaluate_logits, pretrain

```python
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)

class NIDSClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim, pretrained_state=None):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_dim, hidden_dim, embed_dim)
        if pretrained_state is not None:
            self.encoder.load_state_dict(pretrained_state)
        self.head = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        return self.head(h).squeeze(-1)

def pos_weight_for(graphs):
    y_all = torch.cat([g.y[g.active] for g in graphs])
    pos = y_all.sum().item()
    neg = len(y_all) - pos
    return torch.tensor([neg / max(pos, 1)], device=DEVICE)

def evaluate_logits(model, graphs):
    model.eval()
    all_logits, all_y, all_active, all_scenario, all_window = [], [], [], [], []
    with torch.no_grad():
        for g in graphs:
            g = g.to(DEVICE)
            logits = model(g.x, g.edge_index)
            all_logits.append(logits.cpu())
            all_y.append(g.y.cpu())
            all_active.append(g.active.cpu())
            all_scenario += [g.scenario] * g.x.shape[0]
            all_window += [g.window_id] * g.x.shape[0]
    return (torch.cat(all_logits), torch.cat(all_y), torch.cat(all_active),
            np.array(all_scenario), np.array(all_window))

def period_macro_f1(model, graphs):
    """Model-selection metric: average F1 ACROSS periods, not pooled across all nodes.
    Pooled F1 is dominated by exfil (>95% of active val nodes), so it was silently ignoring
    chain performance when picking the 'best' checkpoint -- this fixes that."""
    logits, y, active, scenario, window = evaluate_logits(model, graphs)
    pred = (logits > 0).float()
    f1s = []
    for period in ['exfil', 'chain']:
        mask = active.numpy().astype(bool).copy()
        period_mask = np.zeros(len(mask), dtype=bool)
        for sc in set(scenario):
            sc_mask = scenario == sc
            p_lookup = scenario_periods.get(sc, {})
            this_period = np.array([p_lookup.get(int(w), 'clean') == period for w in window[sc_mask]])
            period_mask[sc_ma
```

### Cell 97
- Matching terms: train_model, pretrain

```python
ablation_models = {}
ablation_reports = []

for name, blocks in ABLATION_CONFIGS.items():
    print(f'\n=== {name} (blocks={blocks}) ===')
    graphs = build_split_graphs(blocks)
    model = NIDSClassifier(FEATURE_DIM, HIDDEN_DIM, EMBED_DIM, pretrained_state=ckpt['encoder_state_dict'])
    model = train_model(model, graphs['train'], graphs['val'], tag=name)
    ablation_models[name] = model
    ablation_reports.append(report(model, graphs, name))

ablation_report = pd.concat(ablation_reports, ignore_index=True)
```

### Cell 107
- Matching terms: pretrain, Optuna

```python
import os, pickle, collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_recall_fscore_support
import optuna

OUT_DIR = r"G:\My Drive\nids-capstone\data\ait\processed"
WINDOW_MINUTES = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

with open(os.path.join(OUT_DIR, f'multi_scenario_graph_dataset_w{WINDOW_MINUTES}m.pkl'), 'rb') as f:
    dataset = pickle.load(f)

ckpt = torch.load(os.path.join(OUT_DIR, 'pretrained_encoder.pt'), map_location=DEVICE, weights_only=False)
HIDDEN_DIM, EMBED_DIM, FEATURE_DIM = ckpt['hidden_dim'], ckpt['embed_dim'], ckpt['in_dim']
print('Pretrained encoder (FIXED during HPO):', FEATURE_DIM, '->', HIDDEN_DIM, '->', EMBED_DIM)

EXFIL_LABELS = {'data exfiltration', 'attacker', 'dnsteal', 'dnsteal-received'}
CHAIN_LABELS = {
    'service_scan', 'dirb_scan', 'wpscan', 'dns_brute_force_start', 'host_discover_dmz',
    'host_discover_local', 'online_cracking', 'upload_rce_shell', 'dump_wp_users',
    'check_whoami', 'check_uname_r', 'read_profile', 'check_who', 'check_meminfo', 'check_uname_a',
    'check_user_id', 'check_df', 'check_netstat_nat', 'check_id', 'read_resolv', 'check_netstat_t',
    'list_home', 'check_last', 'list_web_dir', 'check_date', 'list_www', 'check_netstat_l', 'clear',
    'check_wp_config', 'check_ps_a', 'read_passwd', 'check_release', 'check_cpuinfo', 'read_group',
    'check_uptime', 'check_network_config', 'update/command on unassigned port', 'check_pwd', 'list_l',
}

def window_periods(scen):
    lt = scen['label_table']
    has_chain = lt['labels'].apply(lambda labs: any(l in CHAIN_LABELS for l in labs))
    has_exfil = lt['labels'].apply(lambda labs: any(l in EXFIL_LABELS for l in labs))
    chain_windows = set(lt.loc[has_chain, 'window_id'].astype(int))
    exfil_windows = set(lt.loc[has_exfil, 'window_id'].astype(int)) - chain_windows
    periods = {}
    for w in scen['window_ids']:
        w = int(w)
        periods[w] = 'chain' if w in chain_windows else ('exfil' if w in exfil_windows else 'clean')
    return periods

scenario_periods = {sc: window_periods(scen) for sc, scen in dataset['scenarios'].items()}

def node_features_masked(scen, keep_blocks):
    X_flow = scen['X_flow'] if 'flow' in keep_blocks else np.zeros_like(scen['X_fl
```

### Cell 108
- Matching terms: GraphSAGEEncoder, train_model, evaluate_logits, pretrain

```python
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)

class NIDSClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim, pretrained_state=None, head_hidden=64, dropout=0.0):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_dim, hidden_dim, embed_dim)
        if pretrained_state is not None:
            self.encoder.load_state_dict(pretrained_state)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1))
    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        return self.head(h).squeeze(-1)

def pos_weight_for(graphs, scale=1.0):
    y_all = torch.cat([g.y[g.active] for g in graphs])
    pos = y_all.sum().item()
    neg = len(y_all) - pos
    return torch.tensor([scale * neg / max(pos, 1)], device=DEVICE)

def evaluate_logits(model, graphs):
    model.eval()
    all_logits, all_y, all_active, all_scenario, all_window = [], [], [], [], []
    with torch.no_grad():
        for g in graphs:
            g = g.to(DEVICE)
            logits = model(g.x, g.edge_index)
            all_logits.append(logits.cpu())
            all_y.append(g.y.cpu())
            all_active.append(g.active.cpu())
            all_scenario += [g.scenario] * g.x.shape[0]
            all_window += [g.window_id] * g.x.shape[0]
    return (torch.cat(all_logits), torch.cat(all_y), torch.cat(all_active),
            np.array(all_scenario), np.array(all_window))

def period_macro_f1(model, graphs):
    logits, y, active, scenario, window = evaluate_logits(model, graphs)
    pred = (logits > 0).float()
    f1s = []
    for period in ['exfil', 'chain', 'clean']:
        mask = active.numpy().astype(bool).copy()
        period_mask = np.zeros(len(mask), dtype=bool)
        for sc in set(scenario):
            sc_mask = scenario == sc
            p_lookup = scenario_periods.get(sc, {})
            this_period = np.array([p_lookup.get(int(w), 'clean') == period for w in window[sc_mask]])
            period_mask[sc_mask] = this_period
        m = mask & period_mask
        if m.sum() == 0:
            continue
        yt, yp = y.numpy()[m], pred.numpy()[m]

```

### Cell 109
- Matching terms: train_model, pretrain, Optuna

```python
N_TRIALS = 30

def objective(trial):
    block_choice = trial.suggest_categorical('blocks', list(BLOCK_CHOICES.keys()))
    head_hidden = trial.suggest_categorical('head_hidden', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    pos_weight_scale = trial.suggest_float('pos_weight_scale', 0.5, 2.0)

    graphs = graphs_by_blocks[block_choice]
    model = NIDSClassifier(FEATURE_DIM, HIDDEN_DIM, EMBED_DIM, pretrained_state=ckpt['encoder_state_dict'],
                            head_hidden=head_hidden, dropout=dropout)
    _, best_f1 = train_model(model, graphs['train'], graphs['val'], n_epochs=25, lr=lr,
                              weight_decay=weight_decay, pos_weight_scale=pos_weight_scale,
                              patience=5, tag=f'trial{trial.number}', verbose=False)
    return best_f1

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print('\nBest trial:', study.best_trial.number)
print('Best val macro-F1:', study.best_value)
print('Best params:', study.best_params)
```

### Cell 110
- Matching terms: train_model, evaluate_logits, pretrain, Optuna

```python
best = study.best_params
graphs = graphs_by_blocks[best['blocks']]
final_model = NIDSClassifier(FEATURE_DIM, HIDDEN_DIM, EMBED_DIM, pretrained_state=ckpt['encoder_state_dict'],
                              head_hidden=best['head_hidden'], dropout=best['dropout'])
final_model, final_val_f1 = train_model(
    final_model, graphs['train'], graphs['val'], n_epochs=60, lr=best['lr'],
    weight_decay=best['weight_decay'], pos_weight_scale=best['pos_weight_scale'],
    patience=10, tag='final')
print('Final retrain val macro-F1:', final_val_f1)

def report(model, graphs_by_split, label):
    rows = []
    for split, gs in graphs_by_split.items():
        logits, y, active, scenario, window = evaluate_logits(model, gs)
        pred = (logits > 0).float()
        for period in ['exfil', 'chain', 'clean']:
            mask = active.numpy().astype(bool).copy()
            period_mask = np.zeros(len(mask), dtype=bool)
            for sc in set(scenario):
                sc_mask = scenario == sc
                p_lookup = scenario_periods.get(sc, {})
                this_period = np.array([p_lookup.get(int(w), 'clean') == period for w in window[sc_mask]])
                period_mask[sc_mask] = this_period
            m = mask & period_mask
            if m.sum() == 0:
                continue
            yt, yp = y.numpy()[m], pred.numpy()[m]
            if yt.sum() == 0 and yp.sum() == 0:
                prec = rec = f1 = 1.0
            else:
                prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
            fp = int(((yp == 1) & (yt == 0)).sum())
            n_neg = int((yt == 0).sum())
            rows.append(dict(model=label, split=split, period=period, n=int(m.sum()),
                              n_pos=int(yt.sum()), precision=round(prec,3), recall=round(rec,3),
                              f1=round(f1,3), fpr=round(fp/max(n_neg,1), 4)))
    return pd.DataFrame(rows)

final_report = report(final_model, graphs, f"hpo_tuned_{best['blocks']}")
print(final_report.to_string(index=False))

torch.save(final_model.state_dict(), os.path.join(OUT_DIR, 'final_model_hpo_tuned.pt'))
final_report.to_csv(os.path.join(OUT_DIR, 'phase7_final_report.csv'), index=False)
study.trials_dataframe().to_csv(os.path.join(OUT_DIR, 'phase7_optuna_trials.csv'), index=False)
print('\nSaved final model, report, and all trial results to', OUT_DIR)
```

### Cell 113
- Matching terms: EdgeNIDSClassifier

```python
import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                              precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt

OUT_DIR = r"G:\My Drive\nids-capstone\data\ait\processed"
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
WINDOW_MINUTES = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(OUT_DIR, f'multi_scenario_graph_dataset_w{WINDOW_MINUTES}m.pkl'), 'rb') as f:
    dataset = pickle.load(f)

def build_graphs(scenario_name):
    scen = dataset['scenarios'][scenario_name]
    X = np.concatenate([scen['X_flow'], scen['X_log'], scen['X_emb']], axis=-1)
    graphs = []
    for i, w in enumerate(scen['window_ids']):
        edge_index_np, edge_attr_np = scen['edges'][i]
        data = Data(
            x=torch.tensor(X[i], dtype=torch.float32),
            edge_index=torch.tensor(edge_index_np, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr_np, dtype=torch.float32),
            y=torch.tensor(scen['y'][i], dtype=torch.float32),
            active=torch.tensor(scen['active'][i], dtype=torch.bool),
            num_nodes=X[i].shape[0],
        )
        data.scenario = scenario_name
        data.window_id = int(w)
        graphs.append(data)
    return graphs

all_graphs = {'train': [], 'val': [], 'test': []}
for sc, scen in dataset['scenarios'].items():
    all_graphs[scen['split']] += build_graphs(sc)

FEATURE_DIM = all_graphs['train'][0].x.shape[1]
EDGE_DIM = all_graphs['train'][0].edge_attr.shape[1]

FLOW_NAMES = dataset['scenarios']['russellmitchell']['flow_feature_names']
LOG_NAMES = dataset['scenarios']['russellmitchell']['log_feature_names']
EMB_NAMES = [f'log_emb_{i}' for i in range(dataset['scenarios']['russellmitchell']['X_emb'].shape[-1])]
ALL_FEATURE_NAMES = FLOW_NAMES + LOG_NAMES + EMB_NAMES
assert len(ALL_FEATURE_NAMES) == FEATURE_DIM, 'feature name count mismatch -- check block order'

class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels, edge_dim, out_channels):
        super().__init__(aggr='mean')
        self.lin_msg = nn.Linear(in_channels + edge_dim, out_channels)
        self.lin_root = nn.Linear(in_channels, out_channels)
    def forward(self, x
```

### Cell 121
- Matching terms: GraphSAGEEncoder, GraphMAE, pretrain

```python
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)

class GraphMAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        self.remask_token = nn.Parameter(torch.zeros(embed_dim))
        self.encoder = GraphSAGEEncoder(in_dim, hidden_dim, embed_dim)
        self.decoder = SAGEConv(embed_dim, in_dim)

    def forward(self, x, edge_index, active, mask_rate=0.5, gamma=2.0):
        active_idx = active.nonzero(as_tuple=True)[0]
        if len(active_idx) < 2:
            return None, None
        n_mask = max(1, int(len(active_idx) * mask_rate))
        perm = active_idx[torch.randperm(len(active_idx), device=x.device)]
        mask_idx = perm[:n_mask]

        x_corrupt = x.clone()
        x_corrupt[mask_idx] = self.mask_token
        h = self.encoder(x_corrupt, edge_index)
        h_remask = h.clone()
        h_remask[mask_idx] = self.remask_token
        recon = self.decoder(h_remask, edge_index)

        target, pred = x[mask_idx], recon[mask_idx]
        cos = F.cosine_similarity(pred, target, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
        loss = ((1 - cos) ** gamma).mean()
        return loss, h

def pretrain_tower(get_x_fn, get_active_fn, in_dim, hidden_dim, embed_dim, tag, n_epochs=30):
    model = GraphMAE(in_dim, hidden_dim, embed_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(all_graphs['train'], batch_size=32, shuffle=True)
    history = []
    for epoch in range(n_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            batch = batch.to(DEVICE)
            x = get_x_fn(batch)
            active = get_active_fn(batch)
            if active.sum() < 2:
                continue
            opt.zero_grad()
            loss, _ = model(x, batch.edge_index, active)
            if loss is None:
                continue
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        history.append(avg)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            pr
```

### Cell 122
- Matching terms: GraphSAGEEncoder

```python
class GatedFusionClassifier(nn.Module):
    def __init__(self, flow_dim, log_dim, hidden_dim, embed_dim, flow_state=None, log_state=None):
        super().__init__()
        self.flow_encoder = GraphSAGEEncoder(flow_dim, hidden_dim, embed_dim)
        self.log_encoder = GraphSAGEEncoder(log_dim, hidden_dim, embed_dim)
        if flow_state is not None: self.flow_encoder.load_state_dict(flow_state)
        if log_state is not None: self.log_encoder.load_state_dict(log_state)

        self.gate = nn.Sequential(nn.Linear(embed_dim * 2 + 1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_flow, x_log, edge_index, log_active):
        h_flow = self.flow_encoder(x_flow, edge_index)
        h_log = self.log_encoder(x_log, edge_index)
        g = torch.sigmoid(self.gate(torch.cat([h_flow, h_log, log_active.unsqueeze(-1)], dim=-1)))
        h_fused = g * h_log + (1 - g) * h_flow
        return self.head(h_fused).squeeze(-1), g.squeeze(-1)
```

### Cell 123
- Matching terms: train_model, evaluate_logits

```python
def pos_weight_for(graphs):
    y_all = torch.cat([g.y[g.active] for g in graphs])
    pos = y_all.sum().item()
    neg = len(y_all) - pos
    return torch.tensor([neg / max(pos, 1)], device=DEVICE)

def get_log_x(batch):
    return torch.cat([batch.x_log, batch.x_emb], dim=-1)

def evaluate_logits(model, graphs):
    model.eval()
    all_logits, all_y, all_active, all_scenario, all_window, all_gate = [], [], [], [], [], []
    with torch.no_grad():
        for g in graphs:
            g = g.to(DEVICE)
            logits, gate = model(g.x_flow, get_log_x(g), g.edge_index, g.log_active)
            all_logits.append(logits.cpu())
            all_gate.append(gate.cpu())
            all_y.append(g.y.cpu())
            all_active.append(g.active.cpu())
            all_scenario += [g.scenario] * g.x_flow.shape[0]
            all_window += [g.window_id] * g.x_flow.shape[0]
    return (torch.cat(all_logits), torch.cat(all_y), torch.cat(all_active),
            np.array(all_scenario), np.array(all_window), torch.cat(all_gate))

def period_macro_f1(model, graphs):
    logits, y, active, scenario, window, _ = evaluate_logits(model, graphs)
    pred = (logits > 0).float()
    f1s = []
    for period in ['exfil', 'chain']:
        mask = active.numpy().astype(bool).copy()
        period_mask = np.zeros(len(mask), dtype=bool)
        for sc in set(scenario):
            sc_mask = scenario == sc
            p_lookup = scenario_periods.get(sc, {})
            this_period = np.array([p_lookup.get(int(w), 'clean') == period for w in window[sc_mask]])
            period_mask[sc_mask] = this_period
        m = mask & period_mask
        if m.sum() == 0:
            continue
        yt, yp = y.numpy()[m], pred.numpy()[m]
        if yt.sum() == 0 and yp.sum() == 0:
            f1 = 1.0
        else:
            _, _, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0

def train_model(model, train_graphs, val_graphs, n_epochs=40, lr=1e-3, patience=8, tag=''):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(train_graphs))
    loader = DataLoader(train_graphs, batch_size=32, shuffle=True)

    best_val_f1, best_state, bad_epochs = -1, None, 0
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE
```

### Cell 124
- Matching terms: evaluate_logits

```python
def report(model, graphs_by_split, label):
    rows = []
    for split, gs in graphs_by_split.items():
        logits, y, active, scenario, window, gate = evaluate_logits(model, gs)
        pred = (logits > 0).float()
        for period in ['exfil', 'chain', 'clean']:
            mask = active.numpy().astype(bool).copy()
            period_mask = np.zeros(len(mask), dtype=bool)
            for sc in set(scenario):
                sc_mask = scenario == sc
                p_lookup = scenario_periods.get(sc, {})
                this_period = np.array([p_lookup.get(int(w), 'clean') == period for w in window[sc_mask]])
                period_mask[sc_mask] = this_period
            m = mask & period_mask
            if m.sum() == 0:
                continue
            yt, yp = y.numpy()[m], pred.numpy()[m]
            if yt.sum() == 0 and yp.sum() == 0:
                prec = rec = f1 = 1.0
            else:
                prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
            fp = int(((yp == 1) & (yt == 0)).sum())
            n_neg = int((yt == 0).sum())
            rows.append(dict(model=label, split=split, period=period, n=int(m.sum()),
                              n_pos=int(yt.sum()), precision=round(prec,3), recall=round(rec,3),
                              f1=round(f1,3), fpr=round(fp/max(n_neg,1), 4),
                              mean_gate=round(float(gate.numpy()[m].mean()), 3)))
    return pd.DataFrame(rows)

gated_report = report(gated_model, all_graphs, 'gated_fusion')
print(gated_report.to_string(index=False))

try:
    prior = pd.read_csv(os.path.join(OUT_DIR, 'phase6_ablation_report.csv'))
    compare_cols = ['model','split','period','n','n_pos','precision','recall','f1','fpr']
    combined = pd.concat([prior[compare_cols], gated_report[compare_cols]], ignore_index=True)
    print('\n--- chain period, all models ---')
    print(combined[combined.period == 'chain'].to_string(index=False))
except FileNotFoundError:
    pass
```

### Cell 128
- Matching terms: GraphMAE

```python
class EdgeSAGEConv(MessagePassing):
    """GraphSAGE's mean-aggregation + self-term structure, extended so each neighbor's message
    is conditioned on the edge connecting it (the standard edge-conditioned message-passing
    pattern) instead of ignoring edge_attr entirely, which is what every encoder up to this
    point has done."""
    def __init__(self, in_channels, edge_dim, out_channels):
        super().__init__(aggr='mean')
        self.lin_msg = nn.Linear(in_channels + edge_dim, out_channels)
        self.lin_root = nn.Linear(in_channels, out_channels)
    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out + self.lin_root(x)
    def message(self, x_j, edge_attr):
        return F.relu(self.lin_msg(torch.cat([x_j, edge_attr], dim=-1)))

class EdgeSAGEEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = EdgeSAGEConv(in_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeSAGEConv(hidden_dim, edge_dim, out_dim)
    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        return self.conv2(h, edge_index, edge_attr)

class EdgeGraphMAE(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, embed_dim):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(in_dim))
        self.remask_token = nn.Parameter(torch.zeros(embed_dim))
        self.encoder = EdgeSAGEEncoder(in_dim, edge_dim, hidden_dim, embed_dim)
        self.decoder = EdgeSAGEConv(embed_dim, edge_dim, in_dim)

    def forward(self, x, edge_index, edge_attr, active, mask_rate=0.5, gamma=2.0):
        active_idx = active.nonzero(as_tuple=True)[0]
        if len(active_idx) < 2:
            return None, None
        n_mask = max(1, int(len(active_idx) * mask_rate))
        perm = active_idx[torch.randperm(len(active_idx), device=x.device)]
        mask_idx = perm[:n_mask]

        x_corrupt = x.clone()
        x_corrupt[mask_idx] = self.mask_token
        h = self.encoder(x_corrupt, edge_index, edge_attr)
        h_remask = h.clone()
        h_remask[mask_idx] = self.remask_token
        recon = self.decoder(h_remask, edge_index, edge_attr)

        target, pred = x[mask_idx], recon[mask_idx]
        cos = F.cosine_similarity(pred, target, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
        loss = ((1 - cos) ** gamma).mean()
        return loss, h

mae = Edg
```

### Cell 129
- Matching terms: EdgeNIDSClassifier, pretrain

```python
class EdgeNIDSClassifier(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, embed_dim, pretrained_state=None):
        super().__init__()
        self.encoder = EdgeSAGEEncoder(in_dim, edge_dim, hidden_dim, embed_dim)
        if pretrained_state is not None:
            self.encoder.load_state_dict(pretrained_state)
        self.head = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, edge_index, edge_attr):
        h = self.encoder(x, edge_index, edge_attr)
        return self.head(h).squeeze(-1)

```

### Cell 130
- Matching terms: EdgeNIDSClassifier, train_model, evaluate_logits, pretrain

```python
def pos_weight_for(graphs):
    y_all = torch.cat([g.y[g.active] for g in graphs])
    pos = y_all.sum().item()
    neg = len(y_all) - pos
    return torch.tensor([neg / max(pos, 1)], device=DEVICE)

def evaluate_logits(model, graphs):
    model.eval()
    all_logits, all_y, all_active, all_scenario, all_window = [], [], [], [], []
    with torch.no_grad():
        for g in graphs:
            g = g.to(DEVICE)
            logits = model(g.x, g.edge_index, g.edge_attr)
            all_logits.append(logits.cpu())
            all_y.append(g.y.cpu())
            all_active.append(g.active.cpu())
            all_scenario += [g.scenario] * g.x.shape[0]
            all_window += [g.window_id] * g.x.shape[0]
    return (torch.cat(all_logits), torch.cat(all_y), torch.cat(all_active),
            np.array(all_scenario), np.array(all_window))

def period_macro_f1(model, graphs):
    logits, y, active, scenario, window = evaluate_logits(model, graphs)
    pred = (logits > 0).float()
    f1s = []
    for period in ['exfil', 'chain']:
        mask = active.numpy().astype(bool).copy()
        period_mask = np.zeros(len(mask), dtype=bool)
        for sc in set(scenario):
            sc_mask = scenario == sc
            p_lookup = scenario_periods.get(sc, {})
            this_period = np.array([p_lookup.get(int(w), 'clean') == period for w in window[sc_mask]])
            period_mask[sc_mask] = this_period
        m = mask & period_mask
        if m.sum() == 0:
            continue
        yt, yp = y.numpy()[m], pred.numpy()[m]
        if yt.sum() == 0 and yp.sum() == 0:
            f1 = 1.0
        else:
            _, _, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0

def train_model(model, train_graphs, val_graphs, n_epochs=40, lr=1e-3, patience=8, tag=''):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(train_graphs))
    loader = DataLoader(train_graphs, batch_size=32, shuffle=True)

    best_val_f1, best_state, bad_epochs = -1, None, 0
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            if batch.active.sum() == 0:
                continue
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = loss_
```

### Cell 131
- Matching terms: evaluate_logits

```python
def report(model, graphs_by_split, label):
    rows = []
    for split, gs in graphs_by_split.items():
        logits, y, active, scenario, window = evaluate_logits(model, gs)
        pred = (logits > 0).float()
        for period in ['exfil', 'chain', 'clean']:
            mask = active.numpy().astype(bool).copy()
            period_mask = np.zeros(len(mask), dtype=bool)
            for sc in set(scenario):
                sc_mask = scenario == sc
                p_lookup = scenario_periods.get(sc, {})
                this_period = np.array([p_lookup.get(int(w), 'clean') == period for w in window[sc_mask]])
                period_mask[sc_mask] = this_period
            m = mask & period_mask
            if m.sum() == 0:
                continue
            yt, yp = y.numpy()[m], pred.numpy()[m]
            if yt.sum() == 0 and yp.sum() == 0:
                prec = rec = f1 = 1.0
            else:
                prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
            fp = int(((yp == 1) & (yt == 0)).sum())
            n_neg = int((yt == 0).sum())
            rows.append(dict(model=label, split=split, period=period, n=int(m.sum()),
                              n_pos=int(yt.sum()), precision=round(prec,3), recall=round(rec,3),
                              f1=round(f1,3), fpr=round(fp/max(n_neg,1), 4)))
    return pd.DataFrame(rows)

edge_report = report(edge_model, all_graphs, 'edge_sage')
print(edge_report.to_string(index=False))

try:
    prior = pd.read_csv(os.path.join(OUT_DIR, 'phase6_ablation_report.csv'))
    combined = pd.concat([prior, edge_report], ignore_index=True)
    print('\n--- chain period, all models ---')
    print(combined[combined.period == 'chain'].to_string(index=False))
except FileNotFoundError:
    pass

```

### Cell 138
- Matching terms: Wilcoxon

```python
gnn = per_fold_f1('loso_results_edge_sage.csv')['mean']
mlp = per_fold_f1('loso_results_mlp.csv')['mean']
xgb = per_fold_f1('loso_results_xgboost.csv')['mean']

folds = gnn.index
x = np.arange(len(folds)); width = 0.27
fig, ax = plt.subplots(figsize=(7.1, 3.0))
ax.bar(x - width, gnn.values, width, label='GNN (edge-aware, multimodal)', color='#4C72B0')
ax.bar(x, mlp.reindex(folds).values, width, label='MLP (no graph)', color='#DD8452')
ax.bar(x + width, xgb.reindex(folds).values, width, label='XGBoost (no graph)', color='#55A868')
ax.set_xticks(x); ax.set_xticklabels(folds, rotation=20, ha='right')
ax.set_ylabel('Chain-period F1')
ax.set_title('GNN vs. non-graph baselines, same features, per held-out scenario')
ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_gnn_vs_baselines.pdf'), bbox_inches='tight')
plt.show()

# paired significance, printed alongside the figure
w, pw = stats.wilcoxon(gnn.values, mlp.reindex(folds).values)
t, pt = stats.ttest_rel(gnn.values, mlp.reindex(folds).values)
print(f'GNN vs MLP: Wilcoxon p={pw:.4f}, paired t-test p={pt:.4f}')
w, pw = stats.wilcoxon(gnn.values, xgb.reindex(folds).values)
t, pt = stats.ttest_rel(gnn.values, xgb.reindex(folds).values)
print(f'GNN vs XGBoost: Wilcoxon p={pw:.4f}, paired t-test p={pt:.4f}')
```

### Cell 146
- Matching terms: train_model, evaluate_logits

```python
# S0 -- shared setup for every revision cell (no torch needed here).
# Run AFTER your Phase 5 cells 82-86 so set_seed / build_graphs_from_fold / the model classes /
# train_model / macro_f1_for_selection / evaluate_logits already exist in the kernel.
import os, re, pickle, itertools, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, matthews_corrcoef

OUT_DIR = r"G:\My Drive\nids-capstone\data\ait\processed"
WINDOW_MINUTES = 5
SCENARIOS = ['russellmitchell', 'fox', 'santos', 'wheeler', 'wardbeck', 'wilson', 'harrison']
REV_DIR = os.path.join(OUT_DIR, 'revision'); os.makedirs(REV_DIR, exist_ok=True)
MODEL_DIR = os.path.join(REV_DIR, 'models'); os.makedirs(MODEL_DIR, exist_ok=True)
THRESHOLD = 0.5            # sigmoid(logit) > 0.5  <=>  logit > 0 : fixed, never tuned (state this in the paper)

def load_fold(sc):
    with open(os.path.join(OUT_DIR, f'loso_fold_test-{sc}_w{WINDOW_MINUTES}m.pkl'), 'rb') as f:
        return pickle.load(f)

def _metric(yt, yp):
    if yt.sum() == 0 and yp.sum() == 0:
        return 1.0, 1.0, 1.0
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
    return p, r, f1

def block_bootstrap_ci(y_true, y_pred, block_ids, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(block_ids)
    b2i = {b: np.flatnonzero(block_ids == b) for b in ub}
    f1s = []
    for _ in range(n_boot):
        idx = np.concatenate([b2i[b] for b in rng.choice(ub, size=len(ub), replace=True)])
        f1s.append(_metric(y_true[idx], y_pred[idx])[2])
    return float(np.mean(f1s)), float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))

# same stage map as your Cell 144, so numbers stay comparable with the paper's Fig. 5
STAGE_MAP = {
    'reconnaissance': {'online_cracking','service_scan','host_discover_dmz','host_discover_local',
                       'dns_brute_force_start','network_scan','dns_scan','traceroute'},
    'exploitation': {'dirb_scan','foothold','wpscan','attacker_http','dirb','upload_rce_shell',
                     'webshell_cmd','webshell_upload'},
    'exfiltration': {'data exfiltration','attacker','dnsteal','dnsteal-received','dnsteal-dropped'},
}
def stage_of(label):
    for stage, labs in STAGE_MAP.items():
        if label in labs: return stage
    return 'post-exploitation'
STAGES = ['reconnaissance', 'exploitation', 'post-exploitation', 'exfiltration']

def cano
```

### Cell 148
- Matching terms: GraphMAE, EdgeNIDSClassifier, train_model, pretrain, Optuna

```python
# S2 -- generic LOSO runner that (a) supports GCN/GAT/plain/edge models, (b) keeps per-node probabilities, (c) saves model
# weights, (d) optionally tunes with an equal random-search budget, (e) supports label modes and a positive-class-weight scale.
# Needs Phase 5 cells 82-86 (imports, classes, pretrain_encoder, pos_weight_for, macro_f1_for_selection).
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden=128, embed=128):
        super().__init__()
        self.c1, self.c2 = GCNConv(in_dim, hidden), GCNConv(hidden, embed)
        self.head = nn.Sequential(nn.Linear(embed, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.c1(x, edge_index)); h = self.c2(h, edge_index)
        return self.head(h).squeeze(-1)

class GATClassifier(nn.Module):          # uses the 5-d edge features too (GATConv edge_dim)
    def __init__(self, in_dim, edge_dim, hidden=128, embed=128, heads=4):
        super().__init__()
        self.c1 = GATConv(in_dim, hidden // heads, heads=heads, edge_dim=edge_dim)
        self.c2 = GATConv(hidden, embed, heads=1, edge_dim=edge_dim)
        self.head = nn.Sequential(nn.Linear(embed, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, edge_index, edge_attr=None):
        h = F.elu(self.c1(x, edge_index, edge_attr)); h = self.c2(h, edge_index, edge_attr)
        return self.head(h).squeeze(-1)

def make_model(kind, in_dim, edge_dim, pretrained_state=None):
    if kind == 'edge':  return EdgeNIDSClassifier(in_dim, edge_dim, 128, 128, pretrained_state=pretrained_state)
    if kind == 'plain': return NIDSClassifier(in_dim, 128, 128, pretrained_state=pretrained_state)
    if kind == 'gcn':   return GCNClassifier(in_dim)
    if kind == 'gat':   return GATClassifier(in_dim, edge_dim)
    raise ValueError(kind)

def train_model_v2(model, train_graphs, val_graphs, fold, n_epochs=40, lr=1e-3, wd=0.0, pws=1.0, patience=8):
    """Identical to your train_model, plus weight decay and a positive-class-weight multiplier (pws)."""
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(train_graphs) * pws)
    loader = DataLoader(train_graphs, batch_size=
```

### Cell 150
- Matching terms: GraphMAE, pretrain

```python
# S4 -- RUN THE EXPERIMENTS (each call resumes if interrupted; ~same cost as one of your earlier LOSO sweeps per call).
# Order matters: the first call is needed by almost everything that follows.

# 1) primary model again, this time KEEPING per-node probabilities + weights (AUPRC/MCC, stage analysis, faithfulness)
preds_primary = run_loso_capture('edge_multi', kind='edge')

# 2) other graph models (Item 3).  GCN/GAT have no GraphMAE pretraining, so compare them with your *scratch* run
#    (loso_results_multimodal_edge_scratch.csv).  'egraphsage_style' = edge-aware SAGE whose node inputs are all zero,
#    i.e. only edge (flow) features reach the node -- the closest node-level analogue of E-GraphSAGE.
run_loso_capture('gcn', kind='gcn', use_pretrained=False)
run_loso_capture('gat', kind='gat', use_pretrained=False)
run_loso_capture('egraphsage_style', kind='edge', keep_blocks=(), use_pretrained=False)

# 3) non-graph models that DO see neighbours (Item 3)
run_tabular('mlp', True, 'mlp_nbr')
run_tabular('xgboost', True, 'xgboost_nbr')

# 4) OPTIONAL equal-tuning arm (Item 3): same random-search budget N for every family, scored on the validation scenario.
#    Only do this if you have the compute; otherwise state in the paper that NO model was tuned (see paper text).
# N = 8
# run_loso_capture('edge_multi_tuned', kind='edge', tune_trials=N)
# run_tabular('mlp', True, 'mlp_nbr_tuned', tune_trials=N);  run_tabular('xgboost', True, 'xgboost_nbr_tuned', tune_trials=N)
# run_tabular('mlp', False, 'mlp_tuned', tune_trials=N);     run_tabular('xgboost', False, 'xgboost_tuned', tune_trials=N)

# 5) headline numbers incl. AUPRC and MCC (Item 4).  Report these next to F1 in the paper.
for tag in ['edge_multi', 'gcn', 'gat', 'egraphsage_style', 'mlp_nbr', 'xgboost_nbr']:
    p = os.path.join(OUT_DIR, f'loso_results_{tag}.csv')
    if os.path.isfile(p):
        print(f'\n{tag}:'); print(headline(pd.read_csv(p)))

```

### Cell 151
- Matching terms: gnn_edge_multi, pretrain, Wilcoxon, holm

```python
# S5 -- paired statistics, rewritten per faculty item 1.  NO training needed; reads your saved loso_results_*.csv
CONFIGS = {
    'gnn_edge_multi':         'loso_results_edge_multi.csv',
    'gnn_edge_flow':          'loso_results_flow_only.csv',
    'gnn_plain_multi':        'loso_results_multimodal_plain.csv',
    'gnn_plain_flow':         'loso_results_flow_only_plain.csv',
    'gnn_edge_log':           'loso_results_log_only.csv',
    'gnn_edge_multi_scratch': 'loso_results_edge_multi_scratch.csv',
    'mlp':                    'loso_results_mlp_plain.csv',
    'xgboost':                'loso_results_xgboost_plain.csv',

    # created by the NEW cells below (skipped automatically if the file does not exist yet)
    'mlp_nbr':                'loso_results_mlp_nbr.csv',
    'xgboost_nbr':            'loso_results_xgboost_nbr.csv',
    'gcn':                    'loso_results_gcn.csv',
    'gat':                    'loso_results_gat.csv',
    'egraphsage_style':       'loso_results_egraphsage_style.csv',
}
# Hypothesis FAMILIES are fixed in advance; Holm is applied inside each family (state this in the paper).
COMPARISONS = [  # (family, A, B)  ->  difference = A - B
    ('A: graph vs non-graph', 'gnn_edge_multi', 'mlp'),
    ('A: graph vs non-graph', 'gnn_edge_multi', 'xgboost'),
    ('A: graph vs non-graph', 'gnn_edge_multi', 'mlp_nbr'),
    ('A: graph vs non-graph', 'gnn_edge_multi', 'xgboost_nbr'),
    ('B: 2x2 design', 'gnn_edge_flow',  'gnn_plain_flow'),
    ('B: 2x2 design', 'gnn_edge_multi', 'gnn_plain_multi'),
    ('B: 2x2 design', 'gnn_edge_multi', 'gnn_edge_flow'),
    ('B: 2x2 design', 'gnn_plain_multi', 'gnn_plain_flow'),
    ('C: log-only', 'gnn_edge_log', 'gnn_edge_multi'),
    ('C: log-only', 'gnn_edge_log', 'gnn_edge_flow'),
    ('D: pretraining', 'gnn_edge_multi', 'gnn_edge_multi_scratch'),
    ('E: other graph models', 'gnn_edge_multi_scratch', 'gcn'),
    ('E: other graph models', 'gnn_edge_multi_scratch', 'gat'),
    ('E: other graph models', 'gnn_edge_multi_scratch', 'egraphsage_style'),
]

def fold_f1(csv, period='chain'):
    df = pd.read_csv(os.path.join(OUT_DIR, csv))
    return df[df.period == period].groupby('test_scenario')['f1'].mean().sort_index()

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

def paired_row(a, b, n_boot=10000, seed=0):
    d =
```

### Cell 152
- Matching terms: pretrain

```python
# S6 -- Items 2 and 4 (partly): main results table with per-fold P/R/F1 + block-bootstrap CI, Table I with CIs,
# all-attack baseline and prevalence.  NO training needed: your saved CSVs already contain the CIs.
def chain_table(csv):
    df = pd.read_csv(os.path.join(OUT_DIR, csv)); df = df[df.period == 'chain']
    g = df.groupby('test_scenario').agg(
        precision=('precision', 'mean'), recall=('recall', 'mean'), f1=('f1', 'mean'), f1_sd_seeds=('f1', 'std'),
        ci_lo=('f1_ci_lo', 'mean'), ci_hi=('f1_ci_hi', 'mean'), n=('n', 'first'), n_pos=('n_pos', 'first'))
    return g.sort_index()

T = {'GNN': chain_table('loso_results_edge_multi.csv'),
     'MLP': chain_table('loso_results_mlp_plain.csv'),
     'XGB': chain_table('loso_results_xgboost_plain.csv')}
base = T['GNN'][['n', 'n_pos']].copy()
base['prevalence'] = base.n_pos / base.n
base['all_attack_f1'] = 2 * base.prevalence / (1 + base.prevalence)     # precision = prevalence, recall = 1
print('=== Attack prevalence and the trivial "predict everything = attack" baseline, chain period ===')
print(base.round(3).to_string())
print('mean all-attack F1 over folds: %.3f   (compare with GNN %.3f)' % (base.all_attack_f1.mean(), T['GNN'].f1.mean()))

def cell(r): return f"{r.f1:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
print('\n=== Main results table (chain period; F1 [95% block-bootstrap CI averaged over 3 seeds]) ===')
main = pd.DataFrame({'pos/n': [f"{int(r.n_pos)}/{int(r.n)}" for r in base.itertuples()],
                     'all-attack F1': base.all_attack_f1.round(3).values}, index=base.index)
for k, t in T.items():
    main[f'{k} P'] = t.precision.round(3); main[f'{k} R'] = t.recall.round(3)
    main[f'{k} F1 [CI]'] = [cell(r) for r in t.itertuples()]
print(main.to_string())
main.to_csv(os.path.join(REV_DIR, 'main_results_table.csv'))

# LaTeX body for a table* -- paste inside your tabular (columns: fold, pos/n, all-attack F1, GNN P R F1[CI], MLP F1[CI], XGB F1[CI])
tex = []
for sc in base.index:
    g, m, x = T['GNN'].loc[sc], T['MLP'].loc[sc], T['XGB'].loc[sc]
    tex.append(f"{sc} & {int(base.loc[sc,'n_pos'])}/{int(base.loc[sc,'n'])} & {base.loc[sc,'all_attack_f1']:.2f} & "
               f"{g.precision:.2f} & {g.recall:.2f} & {g.f1:.2f} [{g.ci_lo:.2f}, {g.ci_hi:.2f}] & "
               f"{m.f1:.2f} [{m.ci_lo:.2f}, {m.ci_hi:.2f}] & {x.f1:.2f} [{x.ci_lo:.2f}, {x.ci_hi:.2f}] \\\\")
tex.append(f"\\midrule Mean & -- & {base.all_attack_f1.mean():.2f} & {T['GNN'].precision.mean():.2f} & {T['GNN'].recall.m
```

### Cell 156
- Matching terms: Wilcoxon

```python
# S10 -- Item 7: deletion (faithfulness) test for the saliency result, on ALL 7 folds, plus block-level ablation to
# reconcile "top feature is a log feature" with "fusion does not help".  Needs S4 step 1 (saved models models/edge_multi_*).
def faithfulness_fold(sc, seed=0, k=10, n_random=20, tag='edge_multi', rng_seed=0):
    fold = load_fold(sc); S = fold['scenarios'][sc]
    names = S['flow_feature_names'] + S['log_feature_names'] + [f'emb_{i}' for i in range(S['X_emb'].shape[-1])]
    nf, nl = len(S['flow_feature_names']), len(S['log_feature_names'])
    block = np.array(['flow'] * nf + ['log'] * nl + ['emb'] * (len(names) - nf - nl))
    wp = {int(w): S['window_periods'][i] for i, w in enumerate(S['window_ids'])}
    te = [g for g in build_graphs_from_fold(fold, 'test') if wp.get(g.window_id) == 'chain']
    in_dim, edge_dim = te[0].x.shape[1], te[0].edge_attr.shape[1]
    model = make_model('edge', in_dim, edge_dim)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{tag}_{sc}_s{seed}.pt'), map_location=DEVICE))
    model = model.to(DEVICE).eval()

    def probs(g, zero_idx=()):
        x = g.x.clone()
        if len(zero_idx): x[:, list(zero_idx)] = 0.0        # 0 in standardised space == training-set mean
        with torch.no_grad():
            return torch.sigmoid(model(x.to(DEVICE), g.edge_index.to(DEVICE), g.edge_attr.to(DEVICE))).cpu().numpy()
    def chain_recall(zero_idx=()):
        tp = pos = 0
        for g in te:
            m = g.active.numpy().astype(bool) & g.y.numpy().astype(bool)
            pos += int(m.sum()); tp += int((m & (probs(g, zero_idx) > THRESHOLD)).sum())
        return tp / max(pos, 1), pos

    sal = np.zeros(in_dim); n_tp = 0
    for g in te:
        m = g.active.numpy().astype(bool) & g.y.numpy().astype(bool); p = probs(g)
        for i in np.flatnonzero(m & (p > THRESHOLD)):
            x = g.x.clone().to(DEVICE).requires_grad_(True)
            out = model(x, g.edge_index.to(DEVICE), g.edge_attr.to(DEVICE))
            grad = torch.autograd.grad(out[i], x)[0]
            sal += (grad[i] * x[i]).abs().detach().cpu().numpy(); n_tp += 1
    sal /= max(n_tp, 1)
    top = np.argsort(sal)[::-1][:k]

    rng = np.random.default_rng(rng_seed); base, n_pos = chain_recall()
    rnd_all, rnd_blk = [], []
    for _ in range(n_random):
        rnd_all.append(chain_recall(rng.choice(in_dim, size=k, replace=False))[0])
        pick = []
        for j in top:
            pool = [q for q in np.flatnonzero(block == bloc
```

### Cell 160
- Matching terms: pretrain

```python
run_loso_capture('edge_multi_scratch', kind='edge', use_pretrained=False)
```

### Cell 161
- Matching terms: Optuna

```python
# S12 -- per-fold AUPRC/MCC/F1 for the re-run and new baselines; original (Optuna) vs re-run (default hyper-params)
tags = ['edge_multi','gcn','gat','egraphsage_style','mlp_nbr','xgboost_nbr']
rows = []
for t in tags:
    d = pd.read_csv(os.path.join(OUT_DIR, f'loso_results_{t}.csv'))
    d = d[d.period=='chain'].groupby('test_scenario')[['f1','precision','recall','auprc','mcc','fpr']].mean().round(3)
    d.insert(0, 'model', t); rows.append(d.reset_index())
perfold = pd.concat(rows)
print(perfold[perfold.model=='edge_multi'].to_string(index=False))
print('\nAUPRC per fold'); print(perfold.pivot(index='test_scenario', columns='model', values='auprc').round(3))
print('\nMCC per fold');   print(perfold.pivot(index='test_scenario', columns='model', values='mcc').round(3))
o = pd.read_csv(os.path.join(OUT_DIR,'loso_results_edge_sage.csv'));  o = o[o.period=='chain'].groupby('test_scenario')['f1'].mean()
r = pd.read_csv(os.path.join(OUT_DIR,'loso_results_edge_multi.csv')); r = r[r.period=='chain'].groupby('test_scenario')['f1'].mean()
cmp = pd.DataFrame({'original_tuned': o, 'rerun_default': r}); cmp['diff'] = cmp.rerun_default - cmp.original_tuned
print('\n', cmp.round(3)); print('mean diff', round(cmp['diff'].mean(),3), '| paired t p =', round(stats.ttest_rel(cmp.rerun_default, cmp.original_tuned).pvalue,3))
```

### Cell 163
- Matching terms: train_model, Optuna

```python
import sys, platform, inspect, torch, torch_geometric, sklearn, scipy, numpy
print('python', sys.version.split()[0], '| torch', torch.__version__, '| PyG', torch_geometric.__version__,
      '| sklearn', sklearn.__version__, '| scipy', scipy.__version__, '| numpy', numpy.__version__)
for m in ('xgboost','optuna','sentence_transformers','pandas'):
    try: print(m, __import__(m).__version__)
    except Exception as e: print(m, 'n/a')
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), '|', round(torch.cuda.get_device_properties(0).total_memory/2**30,1), 'GB')
try:
    import psutil; print('RAM GB:', round(psutil.virtual_memory().total/2**30,1))
except Exception: pass
print('CPU:', platform.processor())
try: print('Optuna best params:', study.best_params, '| best value', study.best_value, '| trials', len(study.trials))
except NameError: print('study not in memory -> paste the printed "Best params:" line from the original Optuna cell')
try: print('hyper-params used in the original LOSO:', best)
except NameError: print('best not in memory')
print(inspect.signature(train_model))
```

### Cell 164
- Matching terms: Wilcoxon

```python
N = 8
run_loso_capture('edge_multi_tuned', kind='edge', tune_trials=N)
run_tabular('mlp', True, 'mlp_nbr_tuned', tune_trials=N)
run_tabular('xgboost', True, 'xgboost_nbr_tuned', tune_trials=N)
for tag in ['edge_multi_tuned','mlp_nbr_tuned','xgboost_nbr_tuned']:
    print(f'\n{tag}:'); print(headline(pd.read_csv(os.path.join(OUT_DIR, f'loso_results_{tag}.csv'))))
def f1s(tag): return fold_f1(f'loso_results_{tag}.csv')
for b_ in ['mlp_nbr_tuned','xgboost_nbr_tuned']:
    d = (f1s('edge_multi_tuned') - f1s(b_)).to_numpy()
    print(b_, 'diff', round(d.mean(),3), f'wins {(d>0).sum()}-{(d<0).sum()}', 'p_t', round(stats.ttest_1samp(d,0).pvalue,4), 'p_W', round(stats.wilcoxon(d).pvalue,4))
```

### Cell 167
- Matching terms: Wilcoxon

```python
tags = ['edge_multi','edge_multi_tuned','mlp_nbr','mlp_nbr_tuned','xgboost_nbr','xgboost_nbr_tuned']
tab = pd.DataFrame({t: fold_f1(f'loso_results_{t}.csv') for t in tags}).round(3)
print(tab); print(tab.mean().round(3))
for b_ in ['mlp_nbr_tuned','xgboost_nbr_tuned']:
    d = (tab['edge_multi_tuned'] - tab[b_]).to_numpy()
    print(b_, 'diff', round(d.mean(),3), f'wins {(d>0).sum()}-{(d<0).sum()}', 'p_t', round(stats.ttest_1samp(d,0).pvalue,4), 'p_W', round(stats.wilcoxon(d).pvalue,4))
```

### Cell 168
- Matching terms: Wilcoxon

```python
def fold_f1(csv, period='chain'):
    d = pd.read_csv(os.path.join(OUT_DIR, csv))
    return d[d.period == period].groupby('test_scenario')['f1'].mean().sort_index()

tags = ['edge_multi','edge_multi_tuned','mlp_nbr','mlp_nbr_tuned','xgboost_nbr','xgboost_nbr_tuned']
tab = pd.DataFrame({t: fold_f1(f'loso_results_{t}.csv') for t in tags}).round(3)
print(tab); print(tab.mean().round(3))
for b_ in ['mlp_nbr_tuned','xgboost_nbr_tuned']:
    d = (tab['edge_multi_tuned'] - tab[b_]).to_numpy()
    print(b_, 'diff', round(d.mean(),3), f'wins {(d>0).sum()}-{(d<0).sum()}', 'p_t', round(stats.ttest_1samp(d,0).pvalue,4), 'p_W', round(stats.wilcoxon(d).pvalue,4))
```
