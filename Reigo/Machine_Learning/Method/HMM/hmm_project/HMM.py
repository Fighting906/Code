import numpy as np
from tqdm import tqdm

#隐马尔可夫模型
class HMM_BIO:
    #param:
    # status 状态集合
    # observe 观测集合
    # (A,B,PI) 状态转移矩阵 发射矩阵 初始状态矩阵
    def __init__(self,A=None,B=None,PI=None):
        self.print=True
        
        
        self.N = 7     # 状态集合有多少元素(状态数BIO)
        self.M = 65535 # 观测集合有多少元素(汉字)
        
        #数字化 状态集合Q 观测集合V 
        self.status_dict={'B-PER': 0, # {'盒子1':0，'盒子2':1，'盒子3':2}
                        'I-PER': 1,
                        'B-LOC': 2,
                        'I-LOC': 3,
                        'B-ORG': 4,
                        'I-ORG': 5,
                        'O': 6}
        self.status=list(self.status_dict.keys())          
        #self.observe_dict={}# {'红':0，'白':1}
        self.Q=np.arange(0,self.N) #状态集合 [0,1,2]
        self.V=np.arange(0,self.M) #观测集合 [0,1]
        
        #初始化 (A,B,PI)
        self.A=A # 状态转移矩阵
        self.B=B # 发射矩阵
        self.PI=PI # 初始状态概率
        
        
        if self.print:
            print('状态集合status',self.status_dict)
            print('观测集合observe',self.observe_dict)
            print('状态集合Q',self.Q)
            print('观测集合V',self.V)
            print()
        
    
    #计算前向概率
    #param
    # o 观测序列
    def calc_foward(self,o):
        print('calc_foward')
        #数字化 观测序列O
        O=np.array([self.observe_dict[x] for x in o])
        
        A=self.A
        B=self.B
        
        if self.print:
            print('观测序列O',O)
        
        #初始化alpha
        alpha=self.PI*(B.T[O[0]])
        if self.print:
            print('alpha0',alpha)
            
        #循环计算alpha
        for i in range(1,len(O)):
            alpha=alpha@A*(B.T[O[i]])
            if self.print:
                print(f'alpha{i}',alpha)
        ret=np.sum(alpha)
        print(f'前向概率:{ret}\n')
        return ret
    
    #计算后向概率
    def calc_backward(self,o):
        print('calc_backward')
        #数字化 观测序列O
        O=np.array([self.observe_dict[x] for x in o])
        
        A=self.A
        B=self.B
        
        if self.print:
            print('观测序列O',O)
        
        #初始化beta
        beta=np.ones(self.N)
        
        #循环计算beta
        for i in range(len(O)-1,0,-1):
            beta=A@(beta*(B.T[O[i]]))
            if self.print:
                print(f'beta{i}',beta)
        
        beta=self.PI*(beta*(B.T[O[0]]))
        if self.print:
            print('beta0',beta)
            
        ret=np.sum(beta)
        print(f'后向概率:{ret}\n')
        return ret
    
    #训练模型(A,B,PI)
    def train(self,data):
        self.calc_Param(data)
        pass
    
    #利用数据data(状态序列+观测序列) 计算参数(A,B,PI) 直接使用统计方法 统计A,B,PI
    #param
    # data: (N,(2,n)) N:数据总条数 
    def calc_Param(self,data):
        self.N = len(self.status)  # 状态集合有多少元素
        self.M = len(self.observe)  # 观测集合有多少元素
        
        A=np.zeros((self.N,self.N)) #(N,N)
        B=np.zeros((self.N,self.M)) #(N,M)
        PI=np.zeros(self.N) #(N,)
        
        for i in range(data.shape[0]):
            #d (2,n)状态和观测序列
            #row0 状态序列
            #row1 观测序列
            d=data[i]
            n=d.shape[1]

            PI[d[0,0]]+=1
            B[d[0,0],d[1,0]]+=1
            
            for j in range(1,n):
                #d[0,i] 当前状态
                #d[0,i-1] 上一次状态
                #[1,i] 当前观测
                A[d[0,j-1],d[0,j]]+=1
                B[d[0,j],d[1,j]]+=1
        A=A/np.sum(A,axis=1,keepdims=True)
        B=B/np.sum(B,axis=1,keepdims=True)
        PI=PI/np.sum(PI)
        print('训练参数结果(A,B,PI)')
        print(f'A:{A}')
        print(f'B:{B}')
        print(f'PI:{PI}')
        self.A=A
        self.B=A
        self.PI=PI
 
    #维特比算法(动态规划)
    #已知(A,B,PI) 和一观测序列o 求解其状态序列的概率最大解
    #param:
    # o 观测序列 [0,1,1,1,1,0] shape(n,)
    def viterbi_t(self,o):
        n=o.shape[0]
        dp=np.zeros((self.N,n)) #shape(状态个数,序列长度)
        A=self.A
        B=self.B
        PI=self.PI
        
        dp[:,0]=PI*B[:,o[0]]
        for i in range(1,n):
            dp[:,i]=np.max(dp[:,i-1]*A.T,axis=1)*B[:,o[i]]
        #check(dp)
        return np.argmax(dp,axis=0)
    
    #计算a,b阵列的准确率
    #param:
    # a 源标签
    # b 目的标签
    def calc_acc(self,label,predict):
        return np.mean(np.equal(label,predict))
    
    #在已有的参数(A,B,PI)下随机生成一组长度为n的 状态和观测序列
    #return:
    # np.ndarray[] shape=(2,n)
    # row0 状态序列
    # row1 观测序列
    def generate_Sequence(self,n):
        PI=self.PI
        A=self.A
        B=self.B
        
        ret=np.zeros((2,n),dtype=np.int32)
        
        #从状态集中以概率P随机选择一个状态
        ret[0,0]=np.random.choice(self.Q,p=PI)
        #从该状态生成一个观测值
        ret[1,0]=np.random.choice(self.V,p=B[ret[0,0]])
        
        
        for i in range(1,n):
            ret[0,i]=np.random.choice(self.Q,p=A[ret[0,i-1]])
            ret[1,i]=np.random.choice(self.V,p=B[ret[0,i]])
        return ret
    
    #生成 N组序列 序列中状态长度为n shape=(N,2,n)
    def generate_Data(self,N,n):
        ret=np.array([])
        for i in range(N):
            a=self.generate_Sequence(n)
            a=a.reshape(1,a.shape[0],a.shape[1])
            if i==0:
                ret=a
            else:
                ret=np.r_[ret,a]
        return ret
    
    #数字化观测序列o 将['red','white','red']转换为[0,1,0]
    def observe_transform(self,o):
        return np.array([self.observe_dict[x] for x in o])

def train_BIO(hmm):
    file_path='corpus/BIO_train.txt'
    with open(file_path,mode='r',encoding='utf-8') as fr:
        lines=fr.readlines()
    for i in range(len(lines)):
        if len(lines[i])==1:
            continue
        else:
            ch,tag=lines.split('\t')

def data_pretreatment(hmm:HMM_BIO):
    file_path='corpus/BIO_train.txt'
    with open(file_path,mode='r',encoding='utf-8') as fr:
        lines=fr.readlines()
    ret=None
    for i in range(len(lines)):
        if len(lines[i])==1:
            continue
        else:
            ch,tag=lines.split('\t')
            new_row=np.array([[hmm.status_dict[tag],ord(ch)]])
            if ret==None:
                ret=new_row
            else:
                ret=np.r_(ret,new_row)
    return ret
    
hmm_bio=HMM_BIO()

print(ret.shape)
