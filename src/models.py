
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptionLayer(nn.Module):
    def __init__(self, match_dim, odds_dim, n_teams, n_leagues, emb_dim=8, hidden_dim=64):
        super(PerceptionLayer, self).__init__()
        
        # Match Features -> LSTM (Simulated as FC here for static input, or actual LSTM if seq)
        # Using MLP for simplicity as data is pre-aggregated 
        self.match_net = nn.Sequential(
            nn.Linear(match_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Odds Features -> MLP
        self.odds_net = nn.Sequential(
            nn.Linear(odds_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Metadata -> Embeddings
        self.team_emb = nn.Embedding(n_teams, emb_dim)
        self.league_emb = nn.Embedding(n_leagues, emb_dim)
        
        self.out_dim = hidden_dim + hidden_dim + (emb_dim * 3) # 2 teams + 1 league
        
    def forward(self, match_feats, odds_feats, meta_feats):
        # match_feats: (B, match_dim)
        h_m = self.match_net(match_feats)
        
        # odds_feats: (B, odds_dim)
        h_o = self.odds_net(odds_feats)
        
        # meta_feats: (B, 3) -> [Div, HomeTeam, AwayTeam]
        league_idx = meta_feats[:, 0].long()
        home_idx = meta_feats[:, 1].long()
        away_idx = meta_feats[:, 2].long()
        
        e_l = self.league_emb(league_idx)
        e_h = self.team_emb(home_idx)
        e_a = self.team_emb(away_idx)
        
        # Concatenate
        h_p = torch.cat([h_m, h_o, e_l, e_h, e_a], dim=1)
        return h_p

class ReasoningLayer(nn.Module):
    def __init__(self, input_dim, num_concepts=4, hidden_dim=64):
        super(ReasoningLayer, self).__init__()
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        
        # Projection to Concepts
        self.concept_proj = nn.Linear(input_dim, num_concepts * hidden_dim)
        
        # Graph Adjacency (Learnable)
        self.adj = nn.Parameter(torch.rand(num_concepts, num_concepts))
        
        # GCN Weight
        self.gcn_weight = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project to concepts: (B, K * H) -> (B, K, H)
        concepts = self.concept_proj(x).view(batch_size, self.num_concepts, self.hidden_dim)
        
        # Normalized Adjacency (Simple Softmax or Sigmoid for stability)
        A = torch.sigmoid(self.adj)
        # Add Self loop approx
        A = A + torch.eye(self.num_concepts).to(x.device)
        
        # D^-0.5 A D^-0.5 (Simplified GCN step)
        # Just creating a stable propagation
        # (B, K, H)
        
        # Message Passing
        # H_new = A @ H @ W
        # (K, K) @ (B, K, H) -> (B, K, H)
        
        weighted_concepts = self.gcn_weight(concepts) # (B, K, H)
        
        # Broadcast A for batch
        A_batch = A.unsqueeze(0).expand(batch_size, -1, -1) # (B, K, K)
        
        reasoned_concepts = torch.bmm(A_batch, weighted_concepts) # (B, K, H)
        
        return F.relu(reasoned_concepts)

class AbstractionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AbstractionLayer, self).__init__()
        # Attention Mechanism
        self.query = nn.Linear(hidden_dim, 1)
        
    def forward(self, concepts):
        # concepts: (B, K, H)
        
        # Attention scores
        # (B, K, 1)
        scores = self.query(concepts)
        weights = F.softmax(scores, dim=1)
        
        # Weighted Sum
        # (B, K, H) * (B, K, 1) -> (B, K, H) -> Sum -> (B, H)
        context = torch.sum(concepts * weights, dim=1)
        
        return context

class HRM_DQN(nn.Module):
    def __init__(self, match_dim, odds_dim, n_teams=500, n_leagues=20, n_actions=7):
        super(HRM_DQN, self).__init__()
        
        self.perception = PerceptionLayer(match_dim, odds_dim, n_teams, n_leagues)
        
        input_dim_reasoning = self.perception.out_dim
        self.reasoning = ReasoningLayer(input_dim_reasoning)
        
        self.abstraction = AbstractionLayer(self.reasoning.hidden_dim)
        
        # Dueling Heads
        feat_dim = self.reasoning.hidden_dim
        
        self.value_stream = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        
    def forward(self, state):
        # split state
        # match: 0-16
        # odds: 16-24
        # meta: 24-27
        
        match_feats = state[:, :16].float()
        odds_feats = state[:, 16:24].float()
        meta_feats = state[:, 24:27].float()
        
        h_p = self.perception(match_feats, odds_feats, meta_feats)
        h_r = self.reasoning(h_p)
        h_a = self.abstraction(h_r)
        
        val = self.value_stream(h_a)
        adv = self.advantage_stream(h_a)
        
        # Dueling Aggregation
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        

        return q_vals

class BaselineDQN(nn.Module):
    def __init__(self, match_dim, odds_dim, n_teams=500, n_leagues=20, n_actions=7, 
                 backbone='mlp', dueling=False):
        super(BaselineDQN, self).__init__()
        
        self.perception = PerceptionLayer(match_dim, odds_dim, n_teams, n_leagues)
        self.backbone_type = backbone
        self.dueling = dueling
        self.input_dim = self.perception.out_dim
        self.hidden_dim = 64
        
        if backbone == 'lstm':
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        else: # mlp
            self.fc = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()
            )
            
        if dueling:
            self.value_stream = nn.Linear(self.hidden_dim, 1)
            self.advantage_stream = nn.Linear(self.hidden_dim, n_actions)
        else:
            self.q_stream = nn.Linear(self.hidden_dim, n_actions)
            
    def forward(self, state):
        match_feats = state[:, :16].float()
        odds_feats = state[:, 16:24].float()
        meta_feats = state[:, 24:27].float()
        
        # (B, Dim)
        x = self.perception(match_feats, odds_feats, meta_feats)
        
        if self.backbone_type == 'lstm':
            # Fake sequence dim: (B, 1, Dim)
            x_seq = x.unsqueeze(1)
            # LSTM out: (B, 1, Hidden)
            lstm_out, _ = self.rnn(x_seq)
            # Flatten: (B, Hidden)
            feat = lstm_out[:, -1, :]
        else:
            feat = self.fc(x)
            
        if self.dueling:
            val = self.value_stream(feat)
            adv = self.advantage_stream(feat)
            q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        else:
            q_vals = self.q_stream(feat)
            
        return q_vals

