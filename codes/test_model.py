import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_MLP()
model.load_model(r'C:\Users\lenovo\Desktop\神经网络与深度学习\PJ1\codes\best_models\mlp2_2.pickle')

test_images_path = r'c:\Users\lenovo\Desktop\神经网络与深度学习\PJ1\codes\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'c:\Users\lenovo\Desktop\神经网络与深度学习\PJ1\codes\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))