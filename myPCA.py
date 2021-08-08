# Anomaly Detection Using PCA with T2 and SPE stats
import pandas as pd
import numpy as np
from scipy.stats import f
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
class myPCA:
    def __init__(self) -> None:
        self.scaler_ = StandardScaler()  #初始化scaler用于归一化
        self.components_= None  #负载矩阵
        self.n_components_=0    #主元个数
        self.explained_variance_ = None #PCA后的方差（前n_components个特征值）
        self.explained_variance_ratio_ = .00 #方差贡献率
        self.lambdas_= None     # PCA后全部的特征值（奇异值）
        self.data_train_pca_ = None  #训练集PCA后
        self.data_te_pca_ = None     #测试集PCA后
        self.train_T2_ = None    #训练集的T2统计量
        self.te_T2_ = None     #测试集的T2统计量
        self.train_SPE_ = None   #训练集的SPE统计量
        self.te_SPE_ = None    #测试集的SPE统计量
        self.T2_limit_ = .00    #根据训练集确定的T2统计量
        self.SPE_limit_ = .00   #根据测试集确定的SPE统计量
        pass
    # 读入.csv文件转化为numpy矩阵
    def raw_data(self,file):
        data = pd.read_csv(file)
        data = np.array(data)
        data = np.delete(data,0,1)
        return data
    # 归一化后数据用特征值分解进行PCA
    def myPCA_fit(self, data, contri):
        n=data.shape[0]
        # 得到协方差矩阵S
        S=np.matmul((np.transpose(data)),data/(n-1))
        # 特征值分解得到S的特征值和特征向量
        self.lambdas_,v=np.linalg.eig(S)
        sum_lamb=sum(self.lambdas_)
        contri_sum=0
        i=0
        while contri_sum/sum_lamb<contri:
            contri_sum+=self.lambdas_[i]
            i+=1
        # 得到负载矩阵
        self.components_=v[:,range(i)]
        self.n_components_= i+1
        # 得到主元空间方差
        self.explained_variance_=self.lambdas_[range(i)]
        self.explained_variance_ratio_=contri_sum/sum_lamb
        return 0
    # 根据pca降维后得到的数据集计算T2统计量
    def hotelling_T2(self, train_pca, lambdas):
        T2 = np.array([xi.dot(np.diag(lambdas**-1)).dot(xi.T) for xi in train_pca])
        return T2
    # T2控制限计算
    def T2_limit(self, conf,n,k):
        maxT2 = k * (n**2 - 1)  / (n - k) /n *f.ppf(conf, k, n - k)
        return maxT2
    # 计算SPE（给定PCA前的数据集和负载矩阵p）
    def SPE(self, data,p):
        Q=[]
        n=data.shape[0] # 数据集行数
        m=data.shape[1] # 数据集列数
        for i in range(n):
            temp=np.matmul(data[i],(np.identity(data.shape[1])-p.dot(p.T)))
            temp=np.matmul(temp,data[i].T)
            Q.append(temp)
        return Q
    # SPE控制限theta计算
    def theta_calculation(self, corr_var, n_components, i, m):
        theta = 0
        for k in range(n_components,m):
            theta += (corr_var[k]**i)
        return theta
    # SPE控制限计算
    def SPE_limit(self, p, eigen, n_comp, n_sensors):
        theta1 = self.theta_calculation(eigen, n_comp, 1, n_sensors)
        theta2 = self.theta_calculation(eigen, n_comp, 2, n_sensors)
        theta3 = self.theta_calculation(eigen, n_comp, 3, n_sensors)
        # print(theta1)
        # print(theta2)
        # print(theta3)
        h0 = 1-((2*theta1*theta3)/(3*(theta2**2)))
        c_alpha = norm.ppf(p)
        limit = theta1*((((c_alpha*np.sqrt(2*theta2*(h0**2)))/theta1)+1+(theta2*h0*(h0-1))/(theta1**2))**(1/h0))
        return limit
    # 得到T2统计量后进行plot
    def plot_T2(self, T2 , maxT2 , conf,plt):
        for i in range(len(T2)):
            if T2[i]>=maxT2:
                plt.plot(i, T2[i], 'ro')
            else:
                plt.plot(i, T2[i], 'bo')
        plt.axhline(y = maxT2, color='red')
        plt.xlabel('Samples')
        plt.ylabel('Hotellings T²')
        plt.title('Hotelling T² Statistics Plot- {} % Confidence level'.format(conf*100))
        plt.grid()
        return 0
    # 得到SPE统计量后进行plot
    def plot_SPE(self, SPE , SPE_limit , conf, plt):
        for i in range(len(SPE)):
            if SPE[i]>=SPE_limit:
                plt.plot(i, SPE[i], 'ro')
            else:
                plt.plot(i, SPE[i], 'bo')
        plt.axhline(y = SPE_limit, color='red')
        plt.xlabel('Samples')
        plt.ylabel('Square Prediction Error')
        plt.title('Square Prediction Error Plot - {} % Confidence level'.format(conf*100))
        plt.grid()
        return 0
    # 指定训练集路径和方差贡献率进行PCA训练
    def AD_PCA_train(self, file_train, contri,conf, SVD):
         # *************训练集raw data*************
        data_train = self.raw_data(file_train)
        n = np.shape(data_train)[0] # 保存训练集样本数
        m = np.shape(data_train)[1] # 保存样本在原样本空间特征维度
        # *************初始化scalar对训练集和测试集进行归一化操作*************
        train_scaler = self.scaler_.fit_transform(data_train)
        if SVD==True:
            # *************使用PCA基于SVD对训练集进行降维训练*************
            pca=PCA(n_components=contri)    # 降维后的数据保持contri的信息
            self.data_train_pca_=pca.fit_transform(train_scaler) # 得到训练集降维后n*k的主成分矩阵
            self.components_=pca.components_.T # 负载矩阵m*k
            self.n_components_=np.shape(self.components_)[1] # 主成分个数k
            self.explained_variance_ = pca.explained_variance_ # 特征值向量（从大到小排列共k个）
            self.explained_variance_ratio_ = pca.explained_variance_ratio_  # 累积方差贡献率
            # 计算得到所有特征值以便后续的SPE控制限计算
            pca2=PCA(n_components=m)
            pca2.fit_transform(train_scaler)
            self.lambdas_ = pca2.explained_variance_
            print('The number of principal components:',self.n_components_)
        else:
            # *************使用PCA基于特征值分解对训练集进行降维训练*************
            self.myPCA_fit(train_scaler, contri) # self.lambdas_ 为所有特征值，用于后续的SPE控制限计算
            self.train_pca=np.matmul(train_scaler, self.components_)
            self.n_components_=np.shape(self.components_)[1] # 主成分个数k
        # *************训练集T2统计量*************
        self.train_T2_=self.hotelling_T2(self.data_train_pca_,self.explained_variance_)
        # *************T2控制限计算*************
        self.T2_limit_ = self.T2_limit(conf, n, self.n_components_)
        print('Maximum Confident Limit for T²: {:.2f} ({:.2f} %Confidence)'.format(self.T2_limit_, conf*100))
        # *************训练集SPE统计量计算************
        self.train_SPE_=self.SPE(train_scaler,self.components_)
        # *************SPE控制限计算************
        self.SPE_limit_=self.SPE_limit(conf, self.lambdas_, self.n_components_, m)
        print('Maximum Confident Limit for SPE: {:.2f} ({:.2f} %Confidence)'.format(self.SPE_limit_, conf*100))
        return 0
    # 用训练后的模型来处理测试集数据
    def AD_PCA_trans(self, file_te):
        data_te=self.raw_data(file_te)  # 读入测试集数据
        te_scaler=self.scaler_.transform(data_te)   # 测试集数据归一化
        self.data_te_pca_=np.matmul(te_scaler,self.components_) # 测试集数据PCA
        # *************测试集T2统计量*************
        self.te_T2_=self.hotelling_T2(self.data_te_pca_,self.explained_variance_)
        # *************测试集SPE统计量计算************
        self.te_SPE_=self.SPE(te_scaler,self.components_)
        return 0
    # 顶层函数输入训练集和测试集输出T2统计量图和SPE统计量图
    def AD_PCA(self, file_train, file_te,contri,conf,SVD):
        """Fit the model with file, test the model with file_te
        -------
        self : object
            Returns the instance itself.
        """
        # *************训练集raw data*************
        data_train = self.raw_data(file_train)
        n = np.shape(data_train)[0] # 保存训练集样本数
        m = np.shape(data_train)[1] # 保存样本在原样本空间特征维度
        # *************测试集raw data*************
        data_te = self.raw_data(file_te)
        n_te = np.shape(data_te)[0] # 保存测试集样本数
        # *************初始化scalar对训练集和测试集进行归一化操作*************
        self.scaler_ = StandardScaler()
        train_scaler = self.scaler_.fit_transform(data_train)
        te_scaler = self.scaler_.transform(data_te)# 测试集使用训练集的scaler进行归一化
        self.AD_PCA_train(file_train,contri,conf,SVD)
        self.AD_PCA_trans(file_te)
        # # *************训练集T2统计量*************
        # self.train_T2_=self.hotelling_T2(self.data_train_pca_,self.explained_variance_)
        # # *************测试集T2统计量*************
        # self.te_T2_=self.hotelling_T2(self.data_te_pca_,self.explained_variance_)
        # # *************T2控制限计算*************
        # self.T2_limit_ = self.T2_limit(conf, n, self.n_components_)
        # print('Maximum Confident Limit for T²: {:.2f} ({:.2f} %Confidence)'.format(self.T2_limit_, conf*100))
        # *************T2统计量画图*************
        plt.figure(1)
        self.plot_T2(self.train_T2_, self.T2_limit_, conf, plt)
        plt.figure(2)
        self.plot_T2(self.te_T2_, self.T2_limit_, conf, plt)
        # plt.show()
        # # *************训练集SPE统计量计算************
        # self.train_SPE_=self.SPE(train_scaler,self.components_)
        # # *************测试集SPE统计量计算************
        # self.te_SPE_=self.SPE(te_scaler,self.components_)
        # # *************SPE控制限计算************
        # self.SPE_limit_=self.SPE_limit(conf, self.lambdas_, self.n_components_, m)
        # print('Maximum Confident Limit for SPE: {:.2f} ({:.2f} %Confidence)'.format(self.SPE_limit_, conf*100))
        # *************SPE画图************
        plt.figure(3)
        self.plot_SPE(self.train_SPE_, self.SPE_limit_, conf ,plt)
        plt.figure(4)
        self.plot_SPE(self.te_SPE_, self.SPE_limit_, conf, plt)
        plt.show()
        return 0