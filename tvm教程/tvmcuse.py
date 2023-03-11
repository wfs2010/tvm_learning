# wget onnx model
# wget https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx
# tvmc compile \
# --target "llvm" \
# --input-shapes "data:[1,3,224,224]" \
# --output resnet50-v2-7-tvm.tar \
# resnet50-v2-7.onnx
# 编译后得到的tar解压
# mkdir model
# tar -xvf resnet50-v2-7-tvm.tar -C model
# ls model
# mod.so 是可被 TVM runtime 加载的模型，表示为 C++ 库。
# mod.json 是 TVM Relay 计算图的文本表示。
# mod.params 是包含预训练模型参数的文件。


# 编译模型 --》 tvm runtime
# tvmc compile \
# --target "llvm" \
# --input-shapes "data:[1,3,224,224]" \
# --output resnet50-v2-7-tvm.tar \
# resnet50-v2-7.onnx

# mkdir model
# tar -xvf resnet50-v2-7-tvm.tar -C model
# ls model
# 运行编译模块
# tvmc run \
# --inputs imagenet_cat.npz \
# --output predictions.npz \
# resnet50-v2-7-tvm.tar
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np

img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# 重设大小为 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# ONNX 需要 NCHW 输入, 因此对数组进行转换
img_data = np.transpose(img_data, (2, 0, 1))

# 根据 ImageNet 进行标准化
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_stddev = np.array([0.229, 0.224, 0.225])
norm_img_data = np.zeros(img_data.shape).astype("float32")

for i in range(img_data.shape[0]):
    norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]

# 添加 batch 维度
img_data = np.expand_dims(norm_img_data, axis=0)

# 保存为 .npz（输出 imagenet_cat.npz）
np.savez("imagenet_cat", data=img_data)

import os.path
import numpy as np

from scipy.special import softmax

from tvm.contrib.download import download_testdata

# 下载标签列表
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

output_file = "predictions.npz"

# 打开并读入输出张量
if os.path.exists(output_file):
    with np.load(output_file) as data:
        scores = softmax(data["output_0"])
        scores = np.squeeze(scores)
        ranks = np.argsort(scores)[::-1]

        for rank in ranks[0:5]:
            print("class='%s' with probability=%f" % (labels[rank], scores[rank]))