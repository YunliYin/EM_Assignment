import numpy as np
from scipy import stats
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib as mpl
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
        self.covs = np.zeros((k, d, d))
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
                density[:,i] = norm.pdf(data)
            # 计算所有样本属于每一类别的后验
            posterior = density * self.p
            posterior = posterior / posterior.sum(axis=1, keepdims=True)
            # 计算下一时刻的参数值
            p_hat = posterior.sum(axis=0)
            mean_hat = np.tensordot(posterior, data, axes=[0, 0])
            # 计算协方差
            cov_hat = np.empty(self.covs.shape)
            for i in range(self.K):
                tmp = data - self.means[i]
                cov_hat[i] = np.dot(tmp.T*posterior[:,i], tmp) / p_hat[i]
            # 更新参数
            self.covs = cov_hat
            self.means = mean_hat / p_hat.reshape(-1,1)
            self.p = p_hat / len(data)

        print(self.p)
        print(self.means)
        print(self.covs)

def make_ellipses(mean, cov, ax, confidence=5.991, alpha=0.3, color="blue", eigv=False, arrow_color_list=None):
    """
    多元正态分布
    mean: 均值
    cov: 协方差矩阵
    ax: 画布的Axes对象
    confidence: 置信椭圆置信率 # 置信区间， 95%： 5.991  99%： 9.21  90%： 4.605
    alpha: 椭圆透明度
    eigv: 是否画特征向量
    arrow_color_list: 箭头颜色列表
    """
    lambda_, v = np.linalg.eig(cov)    # 计算特征值lambda_和特征向量v
    # print "lambda: ", lambda_
    # print "v: ", v
    # print "v[0, 0]: ", v[0, 0]

    sqrt_lambda = np.sqrt(np.abs(lambda_))    # 存在负的特征值， 无法开方，取绝对值

    s = confidence
    width = 2 * np.sqrt(s) * sqrt_lambda[0]    # 计算椭圆的两倍长轴
    height = 2 * np.sqrt(s) * sqrt_lambda[1]   # 计算椭圆的两倍短轴
    angle = np.rad2deg(np.arccos(v[0, 0]))    # 计算椭圆的旋转角度
    ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, color=color)    # 绘制椭圆

    ax.add_artist(ell)
    ell.set_alpha(alpha)
    # 是否画出特征向量
    if eigv:
        # print "type(v): ", type(v)
        if arrow_color_list is None:
            arrow_color_list = [color for i in range(v.shape[0])]
        for i in range(v.shape[0]):
            v_i = v[:, i]
            scale_variable = np.sqrt(s) * sqrt_lambda[i]
            # 绘制箭头
            """
            ax.arrow(x, y, dx, dy,    # (x, y)为箭头起始坐标，(dx, dy)为偏移量
                     width,    # 箭头尾部线段宽度
                     length_includes_head,    # 长度是否包含箭头
                     head_width,    # 箭头宽度
                     head_length,    # 箭头长度
                     color,    # 箭头颜色
                     )
            """
            ax.arrow(mean[0], mean[1], scale_variable*v_i[0], scale_variable * v_i[1],
                     width=0.05,
                     length_includes_head=True,
                     head_width=0.2,
                     head_length=0.3,
                     color=arrow_color_list[i])


def plot_2D_gaussian_sampling(mean, cov, ax, data_num=100, confidence=5.991, color="blue", alpha=0.3, eigv=False, data =0):
    """
    mean: 均值
    cov: 协方差矩阵
    ax: Axes对象
    confidence: 置信椭圆的置信率
    data_num: 散点采样数量
    color: 颜色
    alpha: 透明度
    eigv: 是否画特征向量的箭头
    """
    plt.scatter(data[0, :], data[1, :], s=2)
    make_ellipses(mean, cov, ax, confidence=confidence, color=color, alpha=alpha, eigv=eigv)


if __name__ == '__main__':
    dataFile = 'data0.mat'
    data = scio.loadmat(dataFile)
    Firstry=GMM(3,2)
    Firstry.fit(data['data0'].transpose())
    x1, y1 = np.random.multivariate_normal(Firstry.means[0,:].T, Firstry.covs[0], 80).T

    plt.scatter(data['data0'][0,:],data['data0'][1,:],s=2)

    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plot_2D_gaussian_sampling(mean=Firstry.means[0,:], cov=Firstry.covs[0], ax=ax, eigv=True, color="r", data=data['data0'])
    plot_2D_gaussian_sampling(mean=Firstry.means[1, :], cov=Firstry.covs[1], ax=ax, eigv=True, color="g",
                              data=data['data0'])
    plot_2D_gaussian_sampling(mean=Firstry.means[2, :], cov=Firstry.covs[2], ax=ax, eigv=True, color="b",
                              data=data['data0'])
    plt.show()


