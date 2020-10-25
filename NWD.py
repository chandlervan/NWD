import numpy as np
import pandas as pd
import re
from numpy import log,min

class NWD_main:

    def __init__(self, min_count=10, min_support=30, min_s = 3, max_len=4):

        self.min_count = min_count  #录取词语最小出现次数,即词频
        self.min_support = min_support  #录取词语最低支持度，1代表着随机组合,P(AB)/(P(A)*P(B))
        self.min_s = min_s #录取词语最低信息熵，越大说明越有可能独立成词
        self.max_len = max_len #候选词语的最大字数
        self.drop_dict = [u'，', u'\n', u'。', u'、', u'：', u'(', u')', u'[', u']', u'.', 
        u',', u' ', u'\u3000', u'”', u'“', u'？', u'?', u'！', u'‘', u'’', u'…']

        self.t=[] #保存结果用。


    def call(self, s):
        myre = {} #用于快速匹配N字词的正则表达式
        for i in range(2, self.max_len+1):
            #myre = {2:'(..)', 3:'(...)', 4:'(....)', 5:'(.....)', 6:'(......)', 7:'(.......)'}
            myre[i] = '('+ '.'*i +')'

        #定义要去掉的标点字
        for i in self.drop_dict: #去掉标点字
            s = s.replace(i, '')

        # 统计字数
        self.t.append(pd.Series(list(s)).value_counts()) #逐字统计
        tsum = self.t[0].sum() #统计总字数
        rt = [] #保存结果用

        # 生成N字词
        for m in range(2, self.max_len+1):
            print(u'正在生成%s字词...'%m)
            self.t.append([])
            for i in range(m): #生成所有可能的m字词
                self.t[m-1] = self.t[m-1] + re.findall(myre[m], s[i:]) # 这里注意的是为什么用i的遍历，因为要错位生成所有的m字词，i是用于错位的
            
            self.t[m-1] = pd.Series(self.t[m-1]).value_counts() #逐词统计，注意是统计当前字词的次数
            self.t[m-1] = self.t[m-1][self.t[m-1] > self.min_count] #最小次数筛选，这是利用了pandas的筛选方法， t[m-1] > min_count 输出的是一个 true or flase的列表
            tt = self.t[m-1][:] # 复制列表
            
            for k in range(m-1):
                qq = np.array(list(map(lambda ms: tsum*t[m-1][ms]/t[m-2-k][ms[:m-1-k]]/t[k][ms[m-1-k:]], tt.index))) > min_support #最小支持度筛选。
                tt = tt[qq]
            rt.append(tt.index)

        def cal_S(sl): #信息熵计算函数
            return -((sl/sl.sum()).apply(log)*sl/sl.sum()).sum()

        for i in range(2, max_sep+1):
            print(u'正在进行%s字词的最大熵筛选(%s)...'%(i, len(rt[i-2])))
            pp = [] #保存所有的左右邻结果
            for j in range(i+2):
                pp = pp + re.findall('(.)%s(.)'%myre[i], s[j:])
            pp = pd.DataFrame(pp).set_index(1).sort_index() #先排序，这个很重要，可以加快检索速度
            index = np.sort(np.intersect1d(rt[i-2], pp.index)) #作交集
            #下面两句分别是左邻和右邻信息熵筛选
            index = index[np.array(list(map(lambda s: cal_S(pd.Series(pp[0][s]).value_counts()), index))) > min_s]
            rt[i-2] = index[np.array(list(map(lambda s: cal_S(pd.Series(pp[2][s]).value_counts()), index))) > min_s]

        #下面都是输出前处理
        for i in range(len(rt)):
            t[i+1] = t[i+1][rt[i]]
            t[i+1].sort(ascending = False)

        #保存结果并输出
        pd.DataFrame(pd.concat(t[1:])).to_csv('result.txt', header = False)