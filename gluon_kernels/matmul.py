import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl


def test_matmul(backend):
    @triton.jit
    def matmul_triton(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

        offs_k = tl.arange(0, BLOCK_SIZE_K)

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_a = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak

        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_b = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        zero = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=1):
            a = tl.load(a_ptr + offs_a)
            b = tl.load(b_ptr + offs_b)
            accumulator += tl.dot(a, b, acc=zero)

            offs_a += BLOCK_SIZE_K * stride_ak
            offs_b += BLOCK_SIZE_K * stride_bk

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

        tl.store(c_ptr + offs_c, accumulator)


    @gluon.jit
    def matmul_gluon(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M: gl.constexpr,
            BLOCK_SIZE_N: gl.constexpr,
            BLOCK_SIZE_K: gl.constexpr,
    ):
        BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4 ,1], [1, 0])
        MFMA_LAYOUT: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16], True, [2, 2])
        SHARED_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 1, [1, 0])

        pid = gl.program_id(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

        offs_am = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, BLOCKED_LAYOUT))
        offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, BLOCKED_LAYOUT))
        offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak

        offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, BLOCKED_LAYOUT))
        offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, BLOCKED_LAYOUT))
        offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

        accumulator = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=MFMA_LAYOUT)
        steps_a = BLOCK_SIZE_K * stride_ak
        steps_b = BLOCK_SIZE_K * stride_bk
        loop_n = gl.cdiv(K, BLOCK_SIZE_K)

        buffer_a = gl.allocate_shared_memory(gl.float16, (2, BLOCK_SIZE_M, BLOCK_SIZE_K), layout=SHARED_LAYOUT)
        buffer_b = gl.allocate_shared_memory(gl.float16, (2, BLOCK_SIZE_K, BLOCK_SIZE_N), layout=SHARED_LAYOUT)

        # Prologue
        a0 = gl.amd.cdna4.buffer_load(a_ptr, offs_a)
        buffer_a.index(0).store(a0)

        b0 = gl.amd.cdna4.buffer_load(b_ptr, offs_b)
        buffer_b.index(0).store(b0)

        offs_a += steps_a
        offs_b += steps_b

        # Main loop
        for k in range(1, loop_n):
            a = gl.amd.cdna4.buffer_load(a_ptr, offs_a)
            buffer_a.index(k % 2).store(a)

            b = gl.amd.cdna4.buffer_load(b_ptr, offs_b)
            buffer_b.index(k % 2).store(b)

            a = buffer_a.index((k - 1) % 2).load(layout=gl.DotOperandLayout(0, MFMA_LAYOUT, 4))
            b = buffer_b.index((k - 1) % 2).load(layout=gl.DotOperandLayout(1, MFMA_LAYOUT, 4))

            accumulator = gl.amd.cdna4.mfma(a, b, accumulator)

            offs_a += steps_a
            offs_b += steps_b

        # Epilogue
        a = buffer_a.index((loop_n - 1) % 2).load(layout=gl.DotOperandLayout(0, MFMA_LAYOUT, 4))
        b = buffer_b.index((loop_n - 1) % 2).load(layout=gl.DotOperandLayout(1, MFMA_LAYOUT, 4))
        accumulator = gl.amd.cdna4.mfma(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, MFMA_LAYOUT))
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, MFMA_LAYOUT))
        offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

        gl.amd.cdna4.buffer_store(accumulator, c_ptr, offs_c)


    M, N, K = 8192, 8192, 512
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 64

    torch.random.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device='cuda')
    b = torch.randn((K, N), dtype=torch.float16, device='cuda')
    c = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    c_torch = a.to(torch.float32) @ b.to(torch.float32)

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1)
    if backend == 'triton':
        pgm = matmul_triton[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
        torch.testing.assert_close(c, c_torch, rtol=1e-4, atol=1e-4)
    else:
        pgm = matmul_gluon[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
        torch.testing.assert_close(c, c_torch, rtol=1e-4, atol=1e-4)



if __name__ == '__main__':
    test_matmul('triton')
    test_matmul('gluon')
