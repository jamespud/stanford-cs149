import argparse
import numpy as np
import neuronxcc.nki as nki
# 导入你的作业内核
from kernels import (
    vector_add_naive,
    vector_add_tiled,
    vector_add_stream,
    matrix_transpose,
)

def local_simulate_kernel(kernel, *args):
    print(f"\n[本地 CPU 模拟] 正在验证内核: {kernel.__name__}...")

    if kernel == matrix_transpose:
        out_np = args[0].T
        kernel_inputs = [args[0]]
    else:
        out_np = args[0] + args[1]
        kernel_inputs = [args[0], args[1]]

    try:
        out_nki = nki.simulate_kernel(kernel, *kernel_inputs)
    except Exception as e:
        print(f"❌ 模拟执行失败，报错信息:\n{e}")
        return

    if out_nki is None:
        print("❌ 模拟执行没有返回结果，无法校验")
        return

    out_nki = np.asarray(out_nki)
    is_correct = out_nki.shape == out_np.shape and np.allclose(out_nki, out_np, rtol=1e-4, atol=1e-4)
    print(f"验证通过? {'✅ 是 (True)' if is_correct else '❌ 否 (False)'}")
    
    if not is_correct:
        print("\n--- 错误输出对比 ---")
        print("预期形状 (NumPy):", out_np.shape)
        print("实际形状 (NKI):  ", out_nki.shape)
        print("预期结果 (NumPy):", out_np.flatten()[:10])
        print("实际结果 (NKI):  ", out_nki.flatten()[:10])

def main():
    name_to_kernel = {
        "naive": vector_add_naive,
        "tiled": vector_add_tiled,
        "stream": vector_add_stream,
        "transpose": matrix_transpose,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, choices=name_to_kernel.keys(), required=True)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("-m", type=int)
    args = parser.parse_args()

    kernel = name_to_kernel[args.kernel]
    
    # 为了防止 CPU 模拟过慢，建议本地调试时 n 和 m 不要设太大 (比如 128 或 512)
    if kernel == matrix_transpose:
        mat = np.random.rand(args.m or args.n, args.n).astype(np.float32)
        local_simulate_kernel(kernel, mat)
    else:
        a = np.random.rand(args.n).astype(np.float32)
        b = np.random.rand(args.n).astype(np.float32)
        local_simulate_kernel(kernel, a, b)

if __name__ == "__main__":
    main()
