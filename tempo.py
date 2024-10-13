import numpy as np

# 假设preds_sequence是一个NumPy数组
preds_sequence = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 计算每一行的平均值
mean_values = np.mean(preds_sequence, axis=1)

# 将结果转换为列向量
mean_values_column_vector = mean_values.reshape(-1, 1)

print(mean_values_column_vector)




#######这是为了加强落的ALM的调整

 tempo_trends = np.mean(preds_sequence, axis = 1)

    #tempo_trends = preds_sequence[:, args.warming_epochs - 1]
    tempo_trends_mean = tempo_trends.reshape(-1,1)
    final = np.mean(tempo_trends_mean)

    pseudo_targets_1 = np.where(tempo_trends_mean > final, 0, 1)

    print("sadsdddddddddddddddddddddddddd")
    print(pseudo_targets_1.shape)
    pseudo_targets = pseudo_targets_1.squeeze()
    print(pseudo_targets.shape)
    return pseudo_targets