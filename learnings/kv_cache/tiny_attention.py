import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# Set seed
torch.manual_seed(42)


class KVCacheAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Projections for Q, K and V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)


        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=True):
        """
        x shape: (batch, seq_len, d_model)
        """

        bsz, seq_len, _ = x.size()

        # Compute Q, K, V
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)


        if use_cache: 
            if self.cache_k is None:
                self.cache_k = k
                self.cache_v = v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)

            k_to_attend = self.cache_k
            v_to_attend = self.cache_v

        else:
            k_to_attend = k
            v_to_attend = v

        print(f" Step Input Len: {seq_len} | Cache Size After Update: {k_to_attend.shape[1]}")

        scores = torch.matmul(q, k_to_attend.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn = F.softmax(scores, dim=-1)

        output = torch.matmul(attn, v_to_attend)
        return output


if __name__ == "__main__":
    d_model = 4
    model = KVCacheAttention(d_model)
    input_prefills = torch.randn(1, 2, d_model)
    output = model(input_prefills)

    print("Output after prefill:", output)

    st = time.perf_counter()
    for step in range(3):
        input_step = torch.randn(1, 1, d_model)
        output = model(input_step, use_cache=False)
        print(f"Output after step {step+1}:", output)
    print("Time without KV Cache:", time.perf_counter() - st)


    # Simulate step-by-step generation
    st = time.perf_counter()
    for step in range(3):
        input_step = torch.randn(1, 1, d_model)
        output = model(input_step, use_cache=True)
        print(f"Output after step {step+1}:", output)
    print("Time with KV Cache:", time.perf_counter() - st)
    


        