# Most basic HLO snippet of a transformer block

## Assumptions:
* d_model = 256
* d_head = 64
  * num_heads = 4
* Batch size = B
* Sequence length = S
* MLP hidden size = 1024 (4× expansion)
* Ignore dropout, residuals, and layernorm initially to keep it minimal.

## A simplified transformer block in pseudo-HLO
```
ENTRY transformer_block {

  %x = f32[B,S,256] parameter(0)

  // QKV projections
  %Wq = f32[256,256] parameter(1)
  %Wk = f32[256,256] parameter(2)
  %Wv = f32[256,256] parameter(3)

  %Q = dot(%x, %Wq) lhs_contracting_dims={2} rhs_contracting_dims={0}  // [B,S,256]
  %K = dot(%x, %Wk)  // [B,S,256]
  %V = dot(%x, %Wv)  // [B,S,256]

  // Split heads
  %Qh = reshape(%Q)  // [B,S,4,64]
  %Kh = reshape(%K)  // [B,S,4,64]
  %Vh = reshape(%V)  // [B,S,4,64]

  // Move head before sequence
  %Qht = transpose(%Qh), dimensions={0,2,1,3}  // [B,4,S,64]
  %Kht = transpose(%Kh), dimensions={0,2,1,3}  // [B,4,S,64]
  %Vht = transpose(%Vh), dimensions={0,2,1,3}  // [B,4,S,64]

  // Attention scores
  %Scores = dot(%Qht, %Kht) lhs_contracting_dims={3} rhs_contracting_dims={3}  // [B,4,S,S]
  %Scale = constant(0.125)   // 1/sqrt(64)
  %ScaledScores = multiply(%Scores, %Scale)
  %Prob = softmax(%ScaledScores)  // [B,4,S,S]
  %Context = dot(%Prob, %Vht) lhs_contracting_dims={3} rhs_contracting_dims={2}  // [B,4,S,64]
  %ContextT = transpose(%Context), dimensions={0,2,1,3}  // [B,S,4,64]
  %AttnOut = reshape(%ContextT)  // [B,S,256]

  // Output projection
  %Wo = f32[256,256] parameter(4)
  %AttnProj = dot(%AttnOut, %Wo) // [B,S,256]

  // MLP
  %W1 = f32[256,1024] parameter(5)
  %W2 = f32[1024,256] parameter(6)
  %Hidden = dot(%AttnProj, %W1)  // [B,S,1024]
  %Act = gelu(%Hidden)
  %Output = dot(%Act, %W2) // [B,S,256]

  ROOT %Output
}
```

## Fuse the QKV projections into a single dot
```
ENTRY transformer_block {

  %x = f32[B,S,256] parameter(0)

  // Fused QKV projection
  %Wqkv = f32[256,768] parameter(1)

  %QKV = dot(%x, %Wqkv) lhs_contracting_dims={2} rhs_contracting_dims={0} // [B,S,768]

  // Split:  768 = 3 x 4 x 64
  %QKV_reshaped = reshape(%QKV) // [B,S,3,4,64]
  // Slice Q,K,V
  %Q = slice(%QKV_reshaped)  // [B,S,1,4,64]
  %K = slice(%QKV_reshaped)  // [B,S,1,4,64]
  %V = slice(%QKV_reshaped)  // [B,S,1,4,64]

  %Qh = reshape(%Q)  // [B,S,4,64]
  %Kh = reshape(%K)  // [B,S,4,64]
  %Vh = reshape(%V)  // [B,S,4,64]

  // Move head dimension forward
  %Qht = transpose(%Qh), dimensions={0,2,1,3}  // [B,4,S,64]
  %Kht = transpose(%Kh), dimensions={0,2,1,3}  // [B,4,S,64]
  %Vht = transpose(%Vh), dimensions={0,2,1,3}  // [B,4,S,64]

  // Attention scores
  %Scores = dot(%Qht, %Kht) lhs_contracting_dims={3} rhs_contracting_dims={3}  // [B,4,S,S]
  %Scale = constant(0.125)
  %ScaledScores = multiply(%Scores, %Scale)
  %Prob = softmax(%ScaledScores)  // [B,4,S,S]
  %Context = dot(%Prob, %Vht) lhs_contracting_dims={3} rhs_contracting_dims={2}  // [B,4,S,64]
  %ContextT = transpose(%Context), dimensions={0,2,1,3}  // [B,S,4,64]
  %AttnOut = reshape(%ContextT)  // [B,S,256]

  // Output projection
  %Wo = f32[256,256] parameter(2)
  %AttnProj = dot(%AttnOut, %Wo)  // [B,S,256]

  // MLP
  %W1 = f32[256,1024] parameter(3)
  %W2 = f32[1024,256] parameter(4)
  %Hidden = dot(%AttnProj, %W1)  // [B,S,1024]
  %Act = gelu(%Hidden)
  %Output = dot(%Act, %W2)  // [B,S,256]

  ROOT %Output
}
```

## Added RMSNorm 
```
ENTRY transformer_block {

  %x = f32[B,S,256] parameter(0)

  // RMSNorm #1
  %gamma1 = f32[256] parameter(1)
  %sq1 = multiply(%x, %x)  // [B,S,256]
  %mean_sq1 = reduce(%sq1) // [B,S]
  %mean_sq1 = divide(%mean_sq1, constant(256))
  %mean_sq1_b = broadcast(%mean_sq1)  // [B,S,256]
  %eps = constant(1e-6)
  %rms1 = sqrt(add(%mean_sq1_b, %eps))
  %norm1 = divide(%x, %rms1)
  %gamma1_b = broadcast(%gamma1)
  %rmsnorm1 = multiply(%norm1, %gamma1_b)

  // Q projection
  %Wq = f32[256,256] parameter(2)
  %Q = dot(%rmsnorm1, %Wq) lhs_contracting_dims={2} rhs_contracting_dims={0}  // [B,S,256]

  // K projection
  %Wk = f32[256,256] parameter(3)
  %K = dot(%rmsnorm1, %Wk)  // [B,S,256]

  // V projection
  %Wv = f32[256,256] parameter(4)
  %V = dot(%rmsnorm1, %Wv)  // [B,S,256]

  // Split Heads
  %Qh = reshape(%Q)  // [B,S,4,64]
  %Kh = reshape(%K)  // [B,S,4,64]
  %Vh = reshape(%V)  // [B,S,4,64]
  %Qht = transpose(%Qh), dimensions={0,2,1,3}  // [B,4,S,64]
  %Kht = transpose(%Kh), dimensions={0,2,1,3}  // [B,4,S,64]
  %Vht = transpose(%Vh), dimensions={0,2,1,3}  // [B,4,S,64]

  // Attention Scores
  %Scores = dot(%Qht, %Kht) lhs_contracting_dims={3} rhs_contracting_dims={3}  // [B,4,S,S]
  %Scale = constant(0.125)
  %ScaledScores = multiply(%Scores, %Scale)
  %Prob = softmax(%ScaledScores)

  // Attention Output
  %Context = dot(%Prob, %Vht) lhs_contracting_dims={3} rhs_contracting_dims={2}  // [B,4,S,64]
  %ContextT = transpose(%Context),  dimensions={0,2,1,3}
  %AttnOut = reshape(%ContextT)  // [B,S,256]

  // Output Projection
  %Wo = f32[256,256] parameter(5)
  %AttnProj = dot(%AttnOut, %Wo)  // [B,S,256]

  // Residual #1
  %residual1 = add(%x, %AttnProj)  // [B,S,256]
  // RMSNorm #2
  %gamma2 = f32[256] parameter(6)
  %sq2 = multiply(%residual1, %residual1)
  %mean_sq2 = reduce(%sq2)  // [B,S]
  %mean_sq2 = divide(%mean_sq2, constant(256))
  %mean_sq2_b = broadcast(%mean_sq2)
  %rms2 = sqrt(add(%mean_sq2_b, %eps))
  %norm2 = divide(%residual1, %rms2)
  %gamma2_b = broadcast(%gamma2)
  %rmsnorm2 = multiply(%norm2, %gamma2_b)

  // MLP
  %W1 = f32[256,1024] parameter(7)
  %W2 = f32[1024,256] parameter(8)

  %Hidden = dot(%rmsnorm2, %W1)  // [B,S,1024]
  %Act = gelu(%Hidden)
  %MlpOut = dot(%Act, %W2)  // [B,S,256]

  // Residual #2
  %Output =  add(%residual1, %MlpOut)

  ROOT %Output
}
```
