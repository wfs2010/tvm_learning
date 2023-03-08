import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
import torchvision
import tvm
import tvm.testing
import IPython
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from tvm import topi, relax, te
from tvm.script import tir as T

batch_size = 1
input_shape = (batch_size, 1, 28, 28)  # NCHW layout

# Load the weight map from file.
# The prediction accuracy of the weight map on test data is around 83.3%.
# wget -nc https://github.com/mlc-ai/web-data/raw/main/models/fasionmnist_mlp_assignment_params.pkl
weight_map = pkl.load(open("fasionmnist_mlp_assignment_params.pkl", "rb"))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# 第二部分: 从PyTorch迁移模型
def pytorch_model():
    list = []
    list.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), bias=True))
    list.append(nn.ReLU())
    list.append(nn.MaxPool2d(kernel_size=(2, 2)))
    list.append(nn.Flatten())
    list.append(nn.Linear(in_features=5408, out_features=100, bias=True))
    list.append(nn.ReLU())
    list.append(nn.Linear(in_features=100, out_features=10, bias=True))
    list.append(nn.Softmax(dim=1))

    model = nn.Sequential(*list).cpu()
    name_map = {
        "0.weight": "conv2d_weight",
        "0.bias": "conv2d_bias",
        "4.weight": "linear0_weight",
        "4.bias": "linear0_bias",
        "6.weight": "linear1_weight",
        "6.bias": "linear1_bias",
    }
    for name, param in model.named_parameters():
        param.data = torch.from_numpy(weight_map[name_map[name]]).cpu()
    return model


def create_model_via_emit_te():
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo(input_shape, "float32"))

    conv2d_weight = relax.const(weight_map["conv2d_weight"], "float32")
    conv2d_bias = relax.const(weight_map["conv2d_bias"].reshape(1, 32, 1, 1), "float32")
    linear0_weight = relax.const(weight_map["linear0_weight"], "float32")
    linear0_bias = relax.const(weight_map["linear0_bias"].reshape(1, 100), "float32")
    linear1_weight = relax.const(weight_map["linear1_weight"], "float32")
    linear1_bias = relax.const(weight_map["linear1_bias"].reshape(1, 10), "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            c1 = bb.emit_te(tvm.topi.nn.conv2d, x, conv2d_weight, 1, 0, 1)
            c1_bias = bb.emit_te(tvm.topi.add, c1, conv2d_bias)
            relu1 = bb.emit_te(tvm.topi.nn.relu, c1_bias)
            maxpool1 = bb.emit_te(tvm.topi.nn.pool2d, relu1, [2, 2], [2, 2], [1, 1], [0, 0, 0, 0], "max")
            flat1 = bb.emit_te(tvm.topi.nn.flatten, maxpool1)
            l0_w = bb.emit_te(tvm.topi.nn.dense, flat1, linear0_weight)
            l0_bias = bb.emit_te(tvm.topi.add, l0_w, linear0_bias)
            relu2 = bb.emit_te(tvm.topi.nn.relu, l0_bias)
            l1_w = bb.emit_te(tvm.topi.nn.dense, relu2, linear1_weight)
            l1_bias = bb.emit_te(tvm.topi.add, l1_w, linear1_bias)
            output = bb.emit_te(tvm.topi.nn.softmax, l1_bias)
            gv = bb.emit_output(output)

        bb.emit_func_output(gv)

    return bb.get()


def build_mod(mod):
    exec = relax.build(mod, "llvm")
    dev = tvm.cpu()
    vm = relax.VirtualMachine(exec, dev)
    return vm


def check_equivalence(mod, torch_model, test_loader):
    torch_model.eval()
    with torch.no_grad():
        rt_mod = build_mod(mod)
        for data, label in test_loader:
            data, label = data.cpu(), label.cpu()
            output_from_pytorch = torch_model(data).numpy()
            output_from_relax = rt_mod["main"](tvm.nd.array(data, tvm.cpu())).numpy()
            tvm.testing.assert_allclose(output_from_pytorch, output_from_relax, rtol=1e-4)
            break


test_data = torchvision.datasets.FashionMNIST(
    "./data",
    download=True,
    train=False,
    transform=transforms.Compose([transforms.ToTensor()])
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

mod = create_model_via_emit_te()
torch_model = pytorch_model()
# check_equivalence(mod, torch_model, test_loader)
# IPython.display.Code(mod.script(), language="python")
print('the second part end')
print("===========================================")
import tvm.script.relax as R


@tvm.register_func("env.conv", override=True)
def torch_conv(x: tvm.nd.NDArray,
               w: tvm.nd.NDArray,
               b: tvm.nd.NDArray,
               out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    b_torch = torch.from_dlpack(b)
    out_torch = torch.from_dlpack(out)
    # z = torch.conv2d(x_torch, w_torch)
    # 这里使用本身就会报错，可能是原始变量没有复制 这里可能是环境的问题 如果出错的话设置一个中间变量
    out_torch = torch.from_dlpack(out)
    out_torch = torch.conv2d(x_torch, w_torch)
    # print(z[0][0][20])
    # print(out_torch[0][0][20])
    torch.add(out_torch, b_torch, out=out_torch)


def create_model_with_torch_func():
    bb = relax.BlockBuilder()

    x = relax.Var("x", relax.TensorStructInfo(input_shape, "float32"))

    conv2d_weight = relax.const(weight_map["conv2d_weight"], "float32")
    conv2d_bias = relax.const(weight_map["conv2d_bias"].reshape(1, 32, 1, 1), "float32")
    linear0_weight = relax.const(weight_map["linear0_weight"], "float32")
    linear0_bias = relax.const(weight_map["linear0_bias"].reshape(1, 100), "float32")
    linear1_weight = relax.const(weight_map["linear1_weight"], "float32")
    linear1_bias = relax.const(weight_map["linear1_bias"].reshape(1, 10), "float32")

    with bb.function("main", [x]):
        with bb.dataflow():
            # c1 = bb.emit_te(tvm.topi.nn.conv2d, x, conv2d_weight, 1, 0, 1)
            # c1_bias = bb.emit_te(tvm.topi.add, c1, conv2d_bias)
            c1_bias = bb.emit(R.call_tir("env.conv", (x, conv2d_weight, conv2d_bias),
                                         relax.TensorStructInfo((1, 32, 26, 26), "float32")))
            relu1 = bb.emit_te(tvm.topi.nn.relu, c1_bias)
            maxpool1 = bb.emit_te(tvm.topi.nn.pool2d, relu1, [2, 2], [2, 2], [1, 1], [0, 0, 0, 0], "max")
            flat1 = bb.emit_te(tvm.topi.nn.flatten, maxpool1)
            l0_w = bb.emit_te(tvm.topi.nn.dense, flat1, linear0_weight)
            l0_bias = bb.emit_te(tvm.topi.add, l0_w, linear0_bias)
            relu2 = bb.emit_te(tvm.topi.nn.relu, l0_bias)
            l1_w = bb.emit_te(tvm.topi.nn.dense, relu2, linear1_weight)
            l1_bias = bb.emit_te(tvm.topi.add, l1_w, linear1_bias)
            output = bb.emit_te(tvm.topi.nn.softmax, l1_bias)
            gv = bb.emit_output(output)
            # gv = bb.emit_output(c1_bias)

        bb.emit_func_output(gv)

    return bb.get()


test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
mod = create_model_with_torch_func()
# 是这个意思
# check_equivalence(mod, torch_model, test_loader)
print('the third part end')
print("===========================================")


@T.prim_func
def before_inline(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


sch = tvm.tir.Schedule(before_inline)
sch.compute_inline(sch.get_block("B"))


@T.prim_func
def before_fuse(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0


sch = tvm.tir.Schedule(before_fuse)
i, j = sch.get_loops(sch.get_block("B"))
sch.fuse(i, j)

mod = create_model_via_emit_te()
sch = tvm.tir.Schedule(mod)

# Step 1. Get blocks
block_pad = sch.get_block(name="pad_temp", func_name="conv2d")
block_conv2d = sch.get_block(name="conv2d_nchw", func_name="conv2d")

# Step 2. Inline the padding block (if exists)
sch.compute_inline(block_pad)

# Step 3. Get loops
nn, ff, yy, xx, rc, ry, rx = sch.get_loops()
sch.blockize()
# Step 4. Organize the loops

# Step 5. decompose reduction

# Step 6. fuse + vectorize / fuse + parallel / fuse + unroll

print(sch.mod["conv2d"].script())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
check_equivalence(sch.mod, torch_model, test_loader)

print('the fourth part end')
