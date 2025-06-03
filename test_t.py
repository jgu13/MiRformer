import torch
import triton
import triton.language as tl

# 定义一个最小的 Triton kernel
@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr,
               stride_a0, stride_a1,
               stride_b0, stride_b1,
               stride_c0, stride_c1,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # 加载 1×K 大小的 a、b 块（这里演示 K 维）
    a = tl.load(a_ptr + tl.arange(0, 1)[:, None] * stride_a0
                         + tl.arange(0, K)[None, :] * stride_a1)
    b = tl.load(b_ptr + tl.arange(0, 1)[:, None] * stride_b0
                         + tl.arange(0, K)[None, :] * stride_b1)
    # 初始化输出标量
    c = tl.zeros([1, 1], dtype=tl.float32)
    # 在 kernel 里trans_b=True
    b = tl.trans(b) 
    c += tl.dot(a, b) #, trans_b=True)
    # 写回结果
    tl.store(c_ptr, c)

# Host 端准备数据并 launch
def test():
    K = 20
    a = torch.randn(1, K, device='cuda', dtype=torch.float32)
    b = torch.randn(1, K, device='cuda', dtype=torch.float32)
    c = torch.empty(1, 1, device='cuda', dtype=torch.float32)

    dot_kernel[(1, 1)](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        M=1, N=1, K=K
    )
    print("Triton dot:", c.item(), " vs torch:", (a @ b.T).item())

if __name__ == "__main__":
    test()

