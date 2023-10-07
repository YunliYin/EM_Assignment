import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class GMM(object):
    def __init__(self, k: int, d: int):
        '''
        k: K值
        d: 样本属性的数量
        '''
        self.K = k
        # 初始化参数
        self.p = np.random.rand(k)
        self.p = self.p / self.p.sum()      # 保证所有p_k的和为1
        self.means = np.random.rand(k, d)
        self.covs = np.zeros((k, d))
        for i in range(k):                  # 随机生成协方差矩阵，必须是半正定矩阵；这里使用单位阵作为初始值
            self.covs[i] = np.eye(d)

    def fit(self, data: np.ndarray):
        '''
        data: 数据矩阵，每一行是一个样本，shape = (N, d)
        '''
        for _ in range(100):
            density = np.empty((len(data), self.K))
            for i in range(self.K):
                # 生成K个概率密度函数并计算对于所有样本的概率密度
                norm = stats.multivariate_normal(self.means[i], self.covs[i])
                density[:,i] = norm.pdf(data)#计算对于当前步的mean和covs得到的高斯分布，data中各个数据的出现概率
            # 计算所有样本属于每一类别的后验
            posterior = density * self.p
            posterior = posterior / posterior.sum(axis=1, keepdims=True)#进行归一化
            # 计算下一时刻的参数值
            p_hat = posterior.sum(axis=0)
            mean_hat = np.tensordot(posterior, data, axes=[0, 0]) #点乘
            # 计算协方差
            cov_hat = np.empty(self.covs.shape)
            for i in range(self.K):
                tmp = data - self.means[i]
                cov_hat[i] = np.dot(tmp.T*posterior[:,i], tmp) / p_hat[i]
            # 更新参数
            self.covs = cov_hat
            self.means = mean_hat / p_hat.reshape(-1)
            self.p = p_hat / len(data)

        # print(self.p)
        # print(self.means)
        # print(self.covs)



Onedim=GMM(2,1)
n=500
Err1=np.zeros(n)
Err2=np.zeros(n)
Err3=np.zeros(n)
X = np.linspace(100,100*n,n)
for i in range(n):
    num1 = 100*(i+1)
    mu1 = 1
    sigma1 = 1
    num2 = 50*(i+1)
    mu2 = 4
    sigma2 = 2
    data1=np.random.normal(loc=mu1, scale=sigma1, size=num1)
    data2=np.random.normal(loc=mu2, scale=sigma2, size=num2)
    data=np.hstack((data1, data2))
    np.random.shuffle(data.T)
    Onedim.fit(data)
    Err1[i] = max(Onedim.p)
    Err1[i] = (0.66666667 - Err1[i]) * (0.66666667 - Err1[i])
    Err2[i] = max(Onedim.means)
    Err2[i] = (4 - Err2[i]) * (4 - Err2[i])
    Err3[i] = max(Onedim.covs)
    Err3[i] = (4 - Err3[i]) * (4 - Err3[i])
    print(i)


plt.plot(X,Err1,label='P')
plt.plot(X,Err2,label='Means')
plt.plot(X,Err3,label='Sigma')
plt.legend()
plt.xlabel('N')
plt.ylabel('MSE')
plt.show()




