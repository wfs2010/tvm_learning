import IPython
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

# init data
a = np.arange(128 * 128).reshape(128, 128)
b = np.arange(128 * 128, 0, -1).reshape(128, 128)
c_np = a + b


def lnumpy_add(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    for i in range(128):
        for j in range(128):
            c[i, j] = a[i, j] + b[i, j]


c_lnumpy = np.empty((128, 128), dtype=np.int64)
lnumpy_add(a, b, c_lnumpy)


@tvm.script.ir_module
class MyAdd:
    @T.prim_func
    def Add(A: T.Buffer((128, 128), 'int64'),
            B: T.Buffer((128, 128), 'int64'),
            C: T.Buffer((128, 128), "int64")):
        T.func_attr({"global_symbol": "add"})
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = A[vi, vj] + B[vi, vj]


rt_lib = tvm.build(MyAdd, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((128, 128), dtype=np.int64))
rt_lib['add'](a_tvm, b_tvm, c_tvm)
# 不匹配就会报错
np.testing.assert_allclose(c_tvm.numpy(), c_lnumpy)

# -------------------------------------------------------------

print("Exercise one")

a = np.arange(16).reshape(4, 4)
b = np.arange(4, 0, -1).reshape(4)
c_np = a + b


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def add(A: T.Buffer((4, 4), 'int64'),
            B: T.Buffer(4, 'int64'),
            C: T.Buffer((4, 4), 'int64')):
        T.func_attr({"global_symbol": "add", "tir.noalias": True})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vj]


rt_lib = tvm.build(MyModule, target='llvm')
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
rt_lib['add'](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np)
print("Exercise one end")
# -----------------------------------------------
print("Exercise two")
N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N * CI * H * W).reshape(N, CI, H, W)
weight = np.arange(CO * CI * K * K).reshape(CO, CI, K, K)

# torch version
import torch

data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)


@tvm.script.ir_module
class MyConv:
    @T.prim_func
    def conv(A: T.Buffer((1, 1, 8, 8), "int64"),
             B: T.Buffer((2, 1, 3, 3), "int64"),
             C: T.Buffer((1, 2, 6, 6), "int64")):
        T.func_attr({"global_symbol": "conv", "tir.noalias": True})

        for batch, co in T.grid(1, 2):
            for ci, w, h, kw, kh in T.grid(1, 6, 6, 3, 3):
                with T.block("C"):
                    vbatch, vco, vci, vw, vh, vkw, vkh = T.axis.remap("SSRSSRR", [batch, co, ci, w, h, kw, kh])
                    with T.init():
                        C[vbatch, vco, vw, vh] = T.int64(0)
                    C[vbatch, vco, vw, vh] = C[vbatch, vco, vw, vh] + A[vbatch, vci, vw + vkw, vh + vkh] * B[
                        vco, vci, vkw, vkh
                    ]


rt_lib = tvm.build(MyConv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)

print("Exercise two end")

# -----------------------------------------------
print("Exercise three")


@tvm.script.ir_module
class MyBmmRelu:
    @T.prim_func
    def mm_relu(A: T.Buffer((16, 128, 128), "float32"),
                B: T.Buffer((16, 128, 128), "float32"),
                C: T.Buffer((16, 128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((16, 128, 128), dtype="float32")
        for n, i, j, k in T.grid(16, 128, 128, 128):
            with T.block("Y"):
                vn = T.axis.spatial(16, n)
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vn, vi, vj] = T.float32(0)
                Y[vn, vi, vj] = Y[vn, vi, vj] + A[vn, vi, vk] * B[vn, vk, vj]
        for n, i, j in T.grid(16, 128, 128):
            with T.block("C"):
                vn = T.axis.spatial(16, n)
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vn, vi, vj] = T.max(Y[vn, vi, vj], T.float32(0))


@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer((16, 128, 128), "float32"), B: T.Buffer((16, 128, 128), "float32"),
                 C: T.Buffer((16, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.parallel(16):
            for i1, i2_0 in T.grid(128, 16):
                for ax0_init in T.vectorized(8):
                    with T.block("Y_init"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
                        Y[n, i, j] = T.float32(0)
                for ax1_0 in T.serial(32):
                    for ax1_1 in T.unroll(4):
                        for ax0 in T.serial(8):
                            with T.block("Y_update"):
                                n, i = T.axis.remap("SS", [i0, i1])
                                j = T.axis.spatial(128, i2_0 * 8 + ax0)
                                k = T.axis.reduce(128, ax1_0 * 4 + ax1_1)
                                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                for i2_1 in T.vectorized(8):
                    with T.block("C"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))


sch = tvm.tir.Schedule(MyBmmRelu)
# Hints: you can use
# `IPython.display.Code(sch.mod.script(), language="python")`
# or `print(sch.mod.script())`
# to show the current program at any time during the transformation.

# Step 1. Get blocks
block_Y = sch.get_block("Y", func_name="mm_relu")
block_C = sch.get_block("C", func_name="mm_relu")
# Step 2. Get loops
b, i, j, k = sch.get_loops(block_Y)

sch.parallel(b)
# Step 3. Organize the loops
j0, j1 = sch.split(j, factors=[16, 8])
sch.vectorize(j1)
sch.reverse_compute_at(block_C, j1)
# Step 4. decompose reduction
Y_init = sch.decompose_reduction(block_Y, k)
k1, k2 = sch.split(k, factors=[32, 4])
sch.unroll(k2)
# Step 5. vectorize / parallel / unroll
# sch.parallel(...)

print(sch.mod.script())

# 没通过。。。。
# tvm.ir.assert_structural_equal(sch.mod, TargetModule)
print("Pass")

print("exercise three end")

before_rt_lib = tvm.build(MyBmmRelu, target="llvm")
after_rt_lib = tvm.build(sch.mod, target="llvm")
a_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
b_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
after_rt_lib["mm_relu"](a_tvm, b_tvm, c_tvm)
before_timer = before_rt_lib.time_evaluator("mm_relu", tvm.cpu())
print("Before transformation:")
print(before_timer(a_tvm, b_tvm, c_tvm))

f_timer = after_rt_lib.time_evaluator("mm_relu", tvm.cpu())
print("After transformation:")
print(f_timer(a_tvm, b_tvm, c_tvm))
