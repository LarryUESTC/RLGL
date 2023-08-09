import matplotlib.pyplot as plt
import numpy as np
import os

def softmax(z):
    """
    计算softmax函数。

    参数：
        z: 一个NumPy数组，表示一个包含m个样本和n个特征的矩阵，形状为(m, n)。

    返回：
        一个NumPy数组，表示softmax函数的结果，形状为(m, n)。
    """
    # 计算指数
    exp_z = np.exp(z)

    # 计算每行的和
    sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)

    # 计算softmax函数
    softmax = exp_z / sum_exp_z

    return softmax

TABLE_NAME = 'RLGL_SEMI_GCN_Cora'
dataset = 'Cora'
npz_dir = 'exp-6-29-seed-0'
filename = '06-29-12-14-48.npz'

file_path = os.path.join('Temp',TABLE_NAME, npz_dir)
data_filename = os.path.join('Temp',TABLE_NAME, '{}.npz'.format(dataset))
data_np = np.load(data_filename)
train_np = np.load(os.path.join(file_path,filename))

adj = data_np['adj']
label = data_np['label']
idx_train = data_np['idx_train']
idx_val = data_np['idx_val']
idx_test = data_np['idx_test']
all_CE_np = train_np['all_CE_np']
all_predict_np =  train_np['all_predict_np']
sf_all_predict = np.array([softmax(x) for x in all_predict_np])
label_sf_all_predict = np.array([x[np.arange(label.shape[0]), label] for x in sf_all_predict])

cheak_data = label_sf_all_predict

plt.matshow(cheak_data)
plt.title('all_CE_np')
plt.show()

plt.matshow(cheak_data[:, idx_train])
plt.title('idx_train_CE_np')
plt.show()

plt.matshow(cheak_data[:, idx_val])
plt.title('idx_val_CE_np')
plt.show()

plt.matshow(cheak_data[:, idx_test])
plt.title('idx_test_CE_np')
plt.show()

# idx_degree_list = [np.where(adj.sum(0) == k)[0] for k in range(1, 6)]
# idx_degree_list.append(np.where(adj.sum(0) > 5)[0])
# i = 1
# for idx in idx_degree_list:
#     plt.matshow(cheak_data[:, idx])
#     plt.title('degree-{}'.format(i))
#     i +=1
#     plt.show()

last_perdict_label = np.argmax(all_predict_np[-1], axis=1)
correct_idx = np.where(last_perdict_label == label)[0]
plt.matshow(cheak_data[:, correct_idx])
plt.title('correct_idx')
plt.show()

last_perdict_label = np.argmax(all_predict_np[-1], axis=1)
incorrect_idx = np.where(last_perdict_label != label)[0]
plt.matshow(cheak_data[:, incorrect_idx])
plt.title('incorrect_idx')
plt.show()





print("finish")