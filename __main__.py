from myPCA import myPCA
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
# 此函数作用为返回path下所有文件名，不包括文件夹
def get_file(path): # 创建一个空列表
    files_all= os.listdir(path)    # 给定路径下包含文件夹和文件的list
    # print(files_all)
    files= []
    for file in files_all:
        if not os.path.isdir(os.path.join(path,file)): # 判断该文件是否是一个文件夹
            f_name= str(file)
            tr= '\\'  # 多增加2个斜杠
            # filename= path+ tr+ f_name
            filename= f_name
            files.append(filename) # 得到所有
    # print(files[-1])
    # print(os.path.isdir(files[-1]))
    return files#返回文件路径
if __name__=="__main__":
    contri=0.9  # 方差贡献率
    conf=0.95   # 统计量图置信度
    path='C:\\Users\\Dell\\Desktop\\DS\\data'   # 数据集路径（包含训练集和测试集）可修改
    files=get_file(path)    # 得到path下所有文件（不包括文件夹）
    print(files)
    file_num=[]
    # 指定保存结果的总文件夹
    res_dir=path+'\\res'
    # 若无文件夹则创建
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    # 扫描文件夹下各个.csv文件
    for file in files:
        file_split=os.path.splitext(file)
        file_name=file_split[0] # 得到去除文件后缀名的文件名
        end_with=file_split[1]  # 得到文件后缀名
        if end_with=='.csv':
            digit=re.findall(r"\d+",file_name)
            underline=re.findall('_te',file_name)
            # 得到res文件夹下的分文件夹用于保存结果
            res_file = res_dir+'\\'+digit[0]
            # 表明是训练集，则开始训练
            if digit not in file_num:
                file_num.append(digit)
                file_train=path+'\\'+file
                # res文件夹下创建分文件夹用于保存结果
                if not os.path.exists(res_file):
                    os.mkdir(res_file)
                # 开始对训练集进行PCA
                mypca=myPCA()
                mypca.AD_PCA_train(file_train,contri,conf,True)
                # 保存pca后的结果输出为.csv
                k=mypca.n_components_
                columns=[]
                for i in range(k):
                    col='pc'+str(i+1)
                    columns.append(col)
                temp=pd.DataFrame(mypca.data_train_pca_, columns=columns)
                temp.to_csv(res_file+'\\'+file_name+'_components'+'.csv')
                # 训练集T2图片
                plt.figure()
                mypca.plot_T2(mypca.train_T2_,mypca.T2_limit_,conf,plt)
                plt.savefig(res_file+'\\'+file_name+'_T2'+'.jpg')
                plt.close()
                # 训练集SPE图片
                plt.figure()
                mypca.plot_SPE(mypca.train_SPE_,mypca.SPE_limit_,conf,plt)
                plt.savefig(res_file+'\\'+file_name+'_SPE'+'.jpg')
                plt.close()
            # 表明是测试集
            else:
                file_te=path+'\\'+file
                mypca.AD_PCA_trans(file_te)
                # 输出测试集进行pca后的结果
                k=mypca.n_components_
                columns=[]
                for i in range(k):
                    col='pc'+str(i+1)
                    columns.append(col)
                temp=pd.DataFrame(mypca.data_train_pca_, columns=columns)
                temp.to_csv(res_file+'\\'+file_name+'_components'+'.csv')
                # 测试集T2图片
                plt.figure()
                mypca.plot_T2(mypca.te_T2_,mypca.T2_limit_,conf,plt)
                plt.savefig(res_file+'\\'+file_name+'_T2'+'.jpg')
                plt.close()
                plt.figure()
                # 测试集SPE图片
                mypca.plot_SPE(mypca.te_SPE_,mypca.SPE_limit_,conf,plt)
                plt.savefig(res_file+'\\'+file_name+'_SPE'+'.jpg')
                plt.close()
            # 非空表明是测试集
            # if underline!=[]:
    # print(file_num)
    # file_train = 'C:/Users/Dell/Desktop/DS/data/d00.csv'
    # file_te = 'C:/Users/Dell/Desktop/DS/data/d00_te.csv'
    # mypca.AD_PCA(file_train,file_te,contri,conf,True)
    # p=mypca.components_
    # print(p.shape)