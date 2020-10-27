import numpy as np
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
from numpy import log,min

class NWD_base:
    '''
    基本上是 https://kexue.fm/archives/3491 中代码的复制，增加了更多的注释
    共同参考的文章是 http://www.matrix67.com/blog/archives/5044
    '''

    def __init__(self, min_count=10, min_support=30, min_s = 3, max_len=4):

        self.min_count = min_count  #录取词语最小出现次数,即词频
        self.min_support = min_support  #录取词语最低支持度，1代表着随机组合,P(AB)/(P(A)*P(B))
        self.min_s = min_s #录取词语最低信息熵，越大说明越有可能独立成词
        self.max_len = max_len #候选词语的最大字数
        self.drop_dict = [u'，', u'\n', u'。', u'、', u'：', u'(', u')', u'[', u']', u'.', 
        u',', u' ', u'\u3000', u'”', u'“', u'？', u'?', u'！', u'‘', u'’', u'…']

        self.t=[] #保存结果用。

    @staticmethod
    def cal_S(sl):  # 信息熵计算函数
        # ∑ -plog(p)
        p_distribute = sl / sl.sum()  # 计算出概率分布
        log_p_distribute = p_distribute.map(log)  # 计算出对数分布
        return -(p_distribute * log_p_distribute).sum()

    def find_words(self, s):
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
            for k in range(m-1): # 这个循环是用来遍历出当前字符串（ms）的各种组合方式，m-1就是当前的词的索引 t[m-1][ms]就是词ms的词频
                qq = np.array(list(map(lambda ms: tsum * self.t[m-1][ms]/(self.t[m-2-k][ms[:m-1-k]]*self.t[k][ms[m-1-k:]]), tt.index))) > self.min_support #最小支持度筛选。
                tt = tt[qq]
            rt.append(tt.index)

        for i in range(2, self.max_len + 1):
            print(u'正在进行%s字词的最大熵筛选(%s)...' % (i, len(rt[i - 2])))
            pp = []  # 保存所有的左右邻结果
            for j in range(i + 2):  # 这里的循环是为了找出所有的左右邻情况，左右邻都只保留一个字即可
                pp = pp + re.findall('(.)%s(.)' % myre[i], s[j:])
            pp = pd.DataFrame(pp).set_index(1).sort_index()  # 先排序，这个很重要，可以加快检索速度
            # rt[i-2]是有效的通过了支持度筛选的词
            index = np.sort(np.intersect1d(rt[i - 2], pp.index))  # 作交集
            # 下面两句分别是左邻和右邻信息熵筛选
            index = index[
                np.array(list(map(lambda s: self.cal_S(pd.Series(pp[0][s]).value_counts()), index))) > self.min_s]  # 计算左信息熵
            index = index[
                np.array(list(map(lambda s: self.cal_S(pd.Series(pp[2][s]).value_counts()), index))) > self.min_s]  # 计算右信息熵
            rt[i - 2] = index

        #下面都是输出前处理
        for i in range(len(rt)):
            self.t[i+1] = self.t[i+1][rt[i]]
            self.t[i+1].sort_values(ascending = False)

        return pd.DataFrame(pd.concat(self.t[1:]))

class NWD_2P:
    '''
    苏神博客    https://kexue.fm/archives/3956
    2P 的意思是 2 parameters 因为这个方法只用到了词频和凝聚度两个参数
    经过初步测试，在小语料情况下base的效果会好一点，毕竟用到的参数多和条件严格一点
    原抄的代码，不过是加了自己的注释和理解，为了自己用起来更顺手，后续如果有能力或者思路的话就做一定的优化吧
    '''
    def __init__(self, min_count=10, min_pmi=0):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.chars, self.pairs = defaultdict(int), defaultdict(int) #如果键不存在，那么就用int函数
                                                                  #初始化一个值，int()的默认结果为0
        self.total = 0.
    def text_filter(self, texts): #预切断句子，以免得到太多无意义（不是中文、英文、数字）的字符串
        for a in tqdm(texts):
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', a): #这个正则表达式匹配的是任意非中文、
                                                              #非英文、非数字，因此它的意思就是用任
                                                              #意非中文、非英文、非数字的字符断开句子
                if t:
                    yield t

    def count(self, texts): #计数函数，计算单字出现频数、相邻两字出现的频数
        for text in self.text_filter(texts):
            self.chars[text[0]] += 1
            for i in range(len(text)-1):
                self.chars[text[i+1]] += 1
                self.pairs[text[i:i+2]] += 1
                self.total += 1
        self.chars = {i:j for i,j in self.chars.items() if j >= self.min_count} #最少频数过滤
        self.pairs = {i:j for i,j in self.pairs.items() if j >= self.min_count} #最少频数过滤
        self.strong_segments = set()
        for i,j in self.pairs.items(): #根据互信息找出比较“密切”的邻字
            _ = log(self.total*j/(self.chars[i[0]]*self.chars[i[1]]))
            if _ >= self.min_pmi:
                self.strong_segments.add(i)

    def find_words(self, texts): #根据前述结果来找词语
        self.words = defaultdict(int)
        for text in self.text_filter(texts):
            s = text[0]
            for i in range(len(text)-1):
                if text[i:i+2] in self.strong_segments: #如果比较“密切”则不断开
                    s += text[i+1]
                else:
                    self.words[s] += 1 #否则断开，前述片段作为一个词来统计
                    s = text[i+1]
            self.words[s] += 1 #最后一个“词”
        self.words = {i:j for i,j in self.words.items() if j >= self.min_count} #最后再次根据频数过滤
        return sorted(self.words.items(), key=lambda x:x[1], reverse=True)
