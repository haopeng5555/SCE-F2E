import torch
import torch.nn as nn
import torch.nn.functional as F


class cNODEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(cNODEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        out = self.fc2(h)
        return out


class ReformerMoEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4):
        super(ReformerMoEEncoder, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gating_weights = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([F.relu(expert(x)) for expert in self.experts], dim=1)
        weighted_sum = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return weighted_sum


class CapsuleGraphEncoder(nn.Module):
    def __init__(self, input_dim, capsule_dim, num_capsules):
        super(CapsuleGraphEncoder, self).__init__()
        self.capsules = nn.ModuleList([nn.Linear(input_dim, capsule_dim) for _ in range(num_capsules)])

    def squash(self, s):
        mag_sq = torch.sum(s**2, dim=-1, keepdim=True)
        mag = torch.sqrt(mag_sq + 1e-8)
        return (mag_sq / (1.0 + mag_sq)) * (s / (mag + 1e-8))

    def forward(self, x):
        u = torch.stack([caps(x) for caps in self.capsules], dim=1)
        v = self.squash(u)
        return torch.mean(v, dim=1)


class FusionLayer(nn.Module):
    def __init__(self, dim):
        super(FusionLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(3) / 3)

    def forward(self, h1, h2, h3):
        weights = F.softmax(self.alpha, dim=0)
        fused = weights[0] * h1 + weights[1] * h2 + weights[2] * h3
        return fused


class CAPMCTCDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(CAPMCTCDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        h = F.relu(self.fc(x))
        logits = self.out(h)
        return logits


class TranslationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, capsule_dim, num_capsules, vocab_size):
        super(TranslationModel, self).__init__()
        self.encoder1 = cNODEEncoder(input_dim, hidden_dim, hidden_dim)
        self.encoder2 = ReformerMoEEncoder(input_dim, hidden_dim)
        self.encoder3 = CapsuleGraphEncoder(input_dim, capsule_dim, num_capsules)
        self.fusion = FusionLayer(hidden_dim)
        self.decoder = CAPMCTCDecoder(hidden_dim, hidden_dim, vocab_size)

    def forward(self, x):
        h1 = self.encoder1(x)
        h2 = self.encoder2(x)
        h3 = self.encoder3(x)
        fused = self.fusion(h1, h2, h3)
        out = self.decoder(fused)
        return out


# Example usage
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    input_dim = 128
    hidden_dim = 256
    capsule_dim = 64
    num_capsules = 8
    vocab_size = 1000

    model = TranslationModel(input_dim, hidden_dim, capsule_dim, num_capsules, vocab_size)

    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expected: (batch_size, vocab_size)
