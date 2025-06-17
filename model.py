import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch.amp import autocast
from transformers import GPT2Tokenizer


class Gate(nn.Module):
    # (As before)
    def __init__(self, model_dim, num_experts, topk=1, score_func="softmax", route_scale=1.0):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.topk = topk
        self.score_func = score_func
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(num_experts, model_dim))
        self.bias = None
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Args:
            x: [N, model_dim]
        Returns:
            topk_vals: [N, topk]
            topk_idx: [N, topk]
            all_scores: [N, num_experts]
        """
        scores = F.linear(x, self.weight, self.bias)  # [N, E]
        if self.score_func == "softmax":
            all_scores = scores.softmax(dim=-1)
        else:
            all_scores = torch.sigmoid(scores)

        # select top-k
        topk_vals, topk_idx = torch.topk(all_scores, self.topk, dim=-1)  # [N, K], [N, K]

        if self.score_func == "sigmoid" and self.topk > 1:
            # normalize among top-k
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # scale
        topk_vals = topk_vals * self.route_scale  # [N, K]
        return topk_vals, topk_idx, all_scores

class Expert(nn.Module):
    # (As before)
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x):
        # x: [n_tokens, model_dim]
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    def __init__(self, model_dim, num_experts, inter_dim, topk=1, aux_loss_coef=0.01, score_fn="softmax"):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.topk = topk
        self.aux_loss_coef = aux_loss_coef
        self.gate = Gate(model_dim, num_experts, topk=topk, score_func=score_fn, route_scale=1.0)
        self.experts = nn.ModuleList([Expert(model_dim, inter_dim) for _ in range(num_experts)])
        # (Optionally: shared expert, etc.)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, model_dim]
        Returns:
            output: [batch_size, seq_len, model_dim]
            aux_loss: scalar tensor or None
        """
        batch_size, seq_len, dim = x.shape
        assert dim == self.model_dim
        x_flat = x.view(-1, dim)  # [N, D], N = batch_size * seq_len
        N = x_flat.size(0)
        E = self.num_experts
        K = self.topk

        # 1. Gating
        topk_vals, topk_idx, all_scores = self.gate(x_flat)
        # topk_vals: [N, K], topk_idx: [N, K], all_scores: [N, E]

        # 2. Dispatch
        # Prepare output buffer for flattened tokens
        # We'll compute y_flat of shape [N, D]
        # Approach: expand x_flat to [N, K, D], flatten to [N*K, D], process, then sum back.
        if K == 1:
            # existing top-1 logic
            y_flat = torch.zeros_like(x_flat)  # [N, D]
            expert_idx = topk_idx.squeeze(-1)    # [N]
            expert_weight = topk_vals.squeeze(-1)  # [N]
            for i in range(E):
                mask = (expert_idx == i)
                if not mask.any():
                    continue
                x_i = x_flat[mask]            # [n_i, D]
                out_i = self.experts[i](x_i)  # [n_i, D]
                w = expert_weight[mask].unsqueeze(-1)  # [n_i, 1]
                y_flat[mask] = out_i * w
        else:
            # top-k > 1
            # 2.1 Expand and flatten
            # x_flat_expanded: [N, K, D]
            x_flat_expanded = x_flat.unsqueeze(1).expand(-1, K, -1)  # [N, K, D]
            # Flatten to [N*K, D]
            x_exp = x_flat_expanded.contiguous().view(-1, dim)       # [N*K, D]
            # Flatten indices and weights: [N*K]
            idx_flat = topk_idx.contiguous().view(-1)   # [N*K]
            weight_flat = topk_vals.contiguous().view(-1)  # [N*K]
            # Prepare output buffer for each pick: [N*K, D]
            y_exp = torch.zeros_like(x_exp)  # [N*K, D]

            # 2.2 Loop over experts
            # For each expert i, process all positions in flattened picks where idx_flat == i
            for i in range(E):
                mask_i = (idx_flat == i)
                if not mask_i.any():
                    continue
                x_i = x_exp[mask_i]         # [num_assigned, D]
                out_i = self.experts[i](x_i)  # [num_assigned, D]
                w_i = weight_flat[mask_i].unsqueeze(-1)  # [num_assigned, 1]
                # Accumulate weighted outputs
                y_exp[mask_i] = out_i * w_i  # [num_assigned, D]

            # 2.3 Sum contributions from K picks
            # y_exp: [N*K, D] -> reshape to [N, K, D]
            y_flat = y_exp.view(N, K, dim).sum(dim=1)  # [N, D]

        # 3. Compute auxiliary load-balancing loss (top-k version)
        aux_loss = None
        if self.aux_loss_coef and E > 1:
            # For top-k, define f_i as fraction of *assignments* to expert i:
            # total assignments = N * K
            # counts_i = number of times expert i appears in topk_idx
            # So f_i = counts_i / (N*K)
            idx_for_counts = topk_idx.view(-1)  # [N*K]
            counts = torch.bincount(idx_for_counts, minlength=E).to(x_flat.dtype)  # [E]
            f = counts / float(N * K)  # [E], sums to 1

            # P_i: average gating probability for expert i across tokens.
            # all_scores: [N, E], average over N
            P = all_scores.mean(dim=0)  # [E]

            # Auxiliary loss: E * sum_i f_i * P_i
            load_bal_loss = float(E) * torch.dot(f, P)  # scalar
            aux_loss = self.aux_loss_coef * load_bal_loss

        # 4. Reshape back to [batch_size, seq_len, D]
        output = y_flat.view(batch_size, seq_len, dim)
        return output, aux_loss

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # split channel dim in half
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(tensor: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
    # tensor: (B, H, T, head_dim) with head_dim = 2*half_dim
    # freq:   (1, 1, T, half_dim)
    B, H, T, head_dim = tensor.shape
    half_dim = head_dim // 2
    # Split tensor into two halves
    t1, t2 = tensor[..., :half_dim], tensor[..., half_dim:]
    cos, sin = freq.cos(), freq.sin()  # each (1,1,T,half_dim)
    # Apply RoPE on each half
    t1_rot = t1 * cos - t2 * sin
    t2_rot = t1 * sin + t2 * cos
    return torch.cat((t1_rot, t2_rot), dim=-1)

class FlashAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, max_len: int, dropout: int = 0.0, kv_caching: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.scaling = embed_dim ** -0.5
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.kv_caching = kv_caching
        self.v_cache = None
        self.k_cache = None
        self.max_len = max_len
        self.cache_index = 0
        head_dim = embed_dim // num_heads
        half_dim = head_dim // 2
        theta = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        # freq_buffer shape: (max_len, half_dim)
        freq_buffer = torch.outer(torch.arange(max_len, dtype=torch.float32), theta)
        self.register_buffer("freq_buffer", freq_buffer.unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x : torch.Tensor):
        B, T_in, D = x.shape
        H, head_dim = self.num_heads, D // self.num_heads

        # project & split
        qkv = self.qkv_proj(x).view(B, T_in, H, 3, head_dim)
        q, k, v = qkv[...,0,:], qkv[...,1,:], qkv[...,2,:]    # each (B, T_in, H, head_dim)
        """
        This section is quite interesting. So let's break it down:
        The goal is to split qkv into q, k, and v tensors.
        So the intruction qkv[...,i,:] means thet I'm accessing to the dimention corresponding to the i-th matrix.
        So qkv[...,0,:] means I'm accessing to the first matrix of the qkv tensor which is the query matrix and so on.

        I personally find this quite interesting since it is a clver way to access the matrices.
        This is way I used it here so I can remember it and use it in the future.
        ....
        """

        q, k, v = [t.permute(0,2,1,3) for t in (q,k,v)]       # now (B, H, T_in, head_dim)

        freq = self.freq_buffer[:, :, :T_in, : (head_dim // 2)].to(q.device)
        q = apply_rope(q, freq)
        k = apply_rope(k, freq)

        # As the name suggests, this is the KV cache part
        if self.kv_caching:
            if self.k_cache is None:
                # Initialize caches
                self.k_cache = torch.zeros(B, H, self.max_len, head_dim, device=x.device)
                self.v_cache = torch.zeros(B, H, self.max_len, head_dim, device=x.device)
                self.cache_index = 0

            end = self.cache_index + T_in
            if end <= self.max_len:
                self.k_cache[:, :, self.cache_index:end, :] = k
                self.v_cache[:, :, self.cache_index:end, :] = v
                self.cache_index = end
            else:
                shift = end - self.max_len
                self.k_cache = torch.roll(self.k_cache, -shift, dims=2)
                self.v_cache = torch.roll(self.v_cache, -shift, dims=2)
                self.k_cache[:, :, -T_in:, :] = k
                self.v_cache[:, :, -T_in:, :] = v
                self.cache_index = self.max_len

            k = self.k_cache[:, :, :self.cache_index, :]
            v = self.v_cache[:, :, :self.cache_index, :]


        T_k = k.size(2)    # total key/value length
        T_q = q.size(2)    # query length

        if x.device.type == 'cuda':
            # This is flash attention, you can also implement FlashAttention2 or FlashAttention3, or even XFormers
            # But since this is a simple implementation, I will remain simple and use FlashAttention
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
                attn = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout.p,
                    is_causal=True,
                    scale=self.scaling
                )
        else:
            # This is a simple implementation of attention since flash attention is not available on CPU
            mask = torch.triu(
                torch.full((T_q, T_k), -1e9, device=x.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0) 

            scores = torch.matmul(q, k.transpose(-2,-1)) * self.scaling 
            scores = scores + mask
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)

        T_out = attn.size(2) 
        attn = attn.permute(0,2,1,3).reshape(B, T_out, D)

        out = self.out_proj(self.dropout(attn))
        return out
        
class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout:int = 0.1):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.w2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, 4 * n_embd, bias=False)

    def forward(self, x : torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self,
                n_heads: int, 
                n_embd: int, 
                max_len: int, 
                index: int,
                num_dense_layers: int = 1, 
                num_expert: int = 8, 
                score_fn: str = 'softmax', 
                top_k: int = 2,
                inter_size: int = 4096, 
                dropout: int = 0.1, 
                kv_caching: bool = False
                ):
        super(Block, self).__init__()
        self.attention = FlashAttention(n_heads, n_embd, max_len, dropout, kv_caching=kv_caching) 
        self.ff = MLP(n_embd, dropout) if index <= num_dense_layers else MoE(
            model_dim=n_embd,
            num_experts=num_expert,
            inter_dim=inter_size,
            topk=top_k,
            score_fn=score_fn
        )
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.n_experts = num_expert
        self.load_balancing_loss = 0.0
        self.moe_enabled = isinstance(self.ff, MoE)


    def forward(self, x: torch.Tensor):
        attn_out = self.attention(self.ln1(x))
        x = x + attn_out
        if isinstance(self.ff, MoE):
            out, aux_loss = self.ff(self.ln2(x))
            if self.training:
                self.load_balancing_loss = aux_loss.item()
            x = x + out
            return x
        else:
            x = x + self.ff(self.ln2(x))
            return x
    
class Transformer(nn.Module):
    def __init__(self,
                n_layers: int, 
                n_heads: int, 
                n_embd: int, 
                vocab_size: int, 
                num_dense_layers: int = 1, 
                num_expert: int = 8, 
                score_fn: str = 'softmax', 
                top_k: int = 2, 
                inter_size: int = 4096,
                max_len:int = 5000, 
                dropout:int = 0.1,
                kv_caching: bool = False):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        
        # Initialize Arguments for MoE layers
        # Calculate the actual number of MoE layers
        num_moe_layers = max(0, n_layers - num_dense_layers)
        
        self.blocks = nn.ModuleList([Block(n_heads,
                                            n_embd, 
                                            max_len,
                                            index=i,
                                            num_dense_layers=num_dense_layers,
                                            num_expert=num_expert,
                                            score_fn=score_fn,
                                            top_k=top_k, 
                                            inter_size=inter_size,
                                            dropout=dropout, 
                                            kv_caching=kv_caching) for i in range(1, n_layers + 1)])
        self.ln_f = RMSNorm(n_embd)
        self.fc_out = nn.Linear(n_embd, vocab_size)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.num_dense_layers = num_dense_layers
        self.num_expert = num_expert
        self.score_fn = score_fn
        self.top_k = top_k
        self.dropout = dropout
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_length = max_len
        self.kv_caching = kv_caching
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights according to module type."""
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            init.ones_(module.weight)

    def forward(self, x : torch.Tensor):
        x = self.embedding(x)
        for block in self.blocks:
            if self.training and x.requires_grad and x.device.type == 'cuda':
                # Use gradient checkpointing for memory efficiency during training
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.ln_f(x)
        logits = self.fc_out(x)
        
        if self.training:
            moe_loss = sum(block.load_balancing_loss for block in self.blocks if block.moe_enabled)
            return logits, moe_loss
        return logits
    
def generate_texts(
        model: Transformer,
        tokenizer: GPT2Tokenizer, 
        prompts: str, 
        gen_len:int = 50, 
        temperature:float = 1.0, 
        device: str = 'cpu', 
        miwd_precision: bool = False):
    """"
    Generate text using the model.
    """
    model.eval()
    model.to(device)
    input_ids = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(gen_len):
            if miwd_precision:
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids)
            else:
                logits = model(input_ids)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated = torch.cat([generated, next_token], dim=1)
            if input_ids.size(1) > model.max_length:
                input_ids = input_ids[:, -model.max_length:]
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text
