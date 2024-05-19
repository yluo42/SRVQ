import torch
import torch.nn as nn
import torch.nn.functional as F
    
# modified from https://github.com/bshall/VectorQuantizedVAE/blob/master/model.py
class SVQ(nn.Module):
    """
    Spherical VQ via standard EMA-based VQ-VAE scheme.
    """
    def __init__(self, num_code, code_dim, decay=0.99, stale_tolerance=100):
        super().__init__()

        self.num_code = num_code
        self.code_dim = code_dim
        self.decay = decay
        self.stale_tolerance = stale_tolerance
        self.eps = torch.finfo(torch.float32).eps

        # unit-norm codebooks
        embedding = torch.empty(num_code, code_dim).normal_()
        embedding = embedding / (embedding.pow(2).sum(-1) + self.eps).sqrt().unsqueeze(-1)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("ema_count", torch.zeros(self.num_code))
        self.register_buffer("stale_counter", torch.zeros(self.num_code))

    def forward(self, input):

        B, N, T = input.shape
        assert N == self.code_dim

        input_detach = input.detach().mT.reshape(B*T, self.code_dim)  # B*T, dim

        # distance
        eu_dis = 2 - 2 * input_detach.mm(self.embedding.T)  # B*T, num_code

        # best codes
        indices = torch.argmin(eu_dis, dim=-1)  # B*T
        quantized = torch.gather(self.embedding, 0, indices.unsqueeze(-1).expand(-1, self.code_dim))  # B*T, dim
        quantized = quantized.reshape(B, T, N).mT  # B, N, T

        if self.training:
            # EMA update for codebook
            encodings = F.one_hot(indices, self.num_code).float()  # B*T, num_code
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)  # num_code

            update_direction = encodings.T.mm(input_detach)  # num_code, dim
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * update_direction  # num_code, dim

            # Laplace smoothing on the counters
            # make sure the denominator will never be zero
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.eps) / (n + self.num_code * self.eps) * n  # num_code

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

            # calculate code usage
            stale_codes = (encodings.sum(0) == 0).float()  # num_code
            self.stale_counter = self.stale_counter * stale_codes + stale_codes

            # random replace codes that haven't been used for a while
            replace_code = (self.stale_counter == self.stale_tolerance).float() # num_code
            if replace_code.max() > 0:
                random_input_idx = torch.randperm(input_detach.shape[0])
                random_input = input_detach[random_input_idx].reshape(input_detach.shape)
                if random_input.shape[0] < self.num_code:
                    random_input = torch.cat([random_input]*(self.num_code // random_input.shape[0] + 1), 0)
                random_input = random_input[:self.num_code]  # num_code, dim

                self.embedding = self.embedding * (1 - replace_code).unsqueeze(-1) + random_input * replace_code.unsqueeze(-1)
                self.ema_weight = self.ema_weight * (1 - replace_code).unsqueeze(-1) + random_input * replace_code.unsqueeze(-1)
                self.ema_count = self.ema_count * (1 - replace_code)
                self.stale_counter = self.stale_counter * (1 - replace_code)

            # unit-norm codebooks
            self.embedding = self.embedding / (self.embedding.pow(2).sum(-1) + self.eps).sqrt().unsqueeze(-1)
            self.ema_weight = self.ema_weight / (self.ema_weight.pow(2).sum(-1) + self.eps).sqrt().unsqueeze(-1)

        return quantized
    
class RotVQ(nn.Module):
    """
    Rotary VQ via Householder transform.
    """
    def __init__(self, num_code, code_dim):
        super().__init__()

        self.num_code = num_code
        self.code_dim = code_dim
        self.eps = torch.finfo(torch.float32).eps

        self.rot_emb = nn.Parameter(torch.randn(num_code, code_dim))

    def forward(self, prev_input, target):

        B, N, T = prev_input.shape
        assert N == self.code_dim

        prev_input = prev_input.mT.reshape(B*T, N)  # B*T, dim
        target = target.mT.reshape(B*T, N)  # B*T, dim

        # rotation matrices
        # a more efficient implementation without explicitly calculating the rotation matrices
        rot_emb = self.rot_emb / (self.rot_emb.pow(2).sum(-1) + self.eps).sqrt().unsqueeze(-1)
        # always contain an identity rotation matrix
        rot_emb = torch.cat([rot_emb[:,:1] * 0., rot_emb[:,1:]], 1)
        eu_dis = 2 - 2 * (target * prev_input).sum(-1).unsqueeze(-1)  # B*T, 1
        eu_dis = eu_dis + 4 * target.mm(rot_emb.T) * prev_input.mm(rot_emb.T)  # B*T, num_code

        # best codes
        indices = torch.argmin(eu_dis, dim=-1)  # B*T
        encodings = F.one_hot(indices, self.num_code).float()  # B*T, num_code
        rot_emb = encodings.mm(rot_emb)  # B*T, dim
        quantized = prev_input - 2 * (prev_input * rot_emb).sum(-1).unsqueeze(-1) * rot_emb
        quantized = quantized.reshape(B, T, N).mT  # B, N, T

        return quantized
    
class Quantizer(nn.Module):
    def __init__(self, code_dim, decay=0.99, stale_tolerance=100, bit=[12]):
        super().__init__()

        self.code_dim = code_dim
        self.eps = torch.finfo(torch.float32).eps

        self.RVQ = nn.ModuleList([])
        for i in range(len(bit)):
            if i == 0:
                self.RVQ.append(SVQ(2**bit[i], code_dim, decay, stale_tolerance))
            else:
                self.RVQ.append(RotVQ(2**bit[i], code_dim))

    def forward(self, input):
        
        quantized = []
        for i in range(len(self.RVQ)):
            if i == 0:
                this_quantized = self.RVQ[i](input)
                # straight-through estimator
                this_quantized = (this_quantized - input).detach() + input
            else:
                this_quantized = self.RVQ[i](quantized[-1], input)
            quantized.append(this_quantized)
        
        latent_loss = []
        for i in range(len(self.RVQ)):
            if i == 0:
                latent_loss.append(F.mse_loss(input, quantized[i].detach()))
            else:
                latent_loss.append((F.mse_loss(input, quantized[i].detach()) + F.mse_loss(input.detach(), quantized[i])) / 2.)

        quantized = torch.stack(quantized, -1)
        latent_loss = torch.stack(latent_loss, -1)

        return quantized, latent_loss

if __name__ == '__main__':
    Q = Quantizer(code_dim=64, bit=[10]*5)
    input = torch.rand(2, 64, 100)
    input = input / input.pow(2).sum(1).sqrt().unsqueeze(1)

    quantized, latent_loss = Q(input)  # no need to apply straight-through estimator between quantized and input again
    print(quantized.shape)
    print(latent_loss)  # non-increasing