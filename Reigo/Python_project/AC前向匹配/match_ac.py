#用ac自动机进行匹配
#暂时未使用日志记录模块中方法

import record_log as reclog
import ac_Tree_forMatch as ac_tree

class ac_match_t():
    def __init__(self,rep_rec=reclog.rep_rec_t()):
        self.ac=ac_tree.Aho_Corasick_automaton() #ac自动机
        self.str_repl={} # 被替换词对应替换词字典
        self.rep_rec = rep_rec  # 替换记录器

        self.parsedTriples=[]  # 经过match解析后的三元组
        self.newParsedTriples=[] # 经过规则选择后的三元组

    #集成化方法(该类的使用方法)
    #调用solve 在当前已有的替换字典下,直接将txt,进行多模式匹配

    #param
    #dict 字典 {被替换词:替换词}
    #txt 字符串 "被识别的字符串"
    def solve(self,dict=None,txt=None):

        #如果没有输入则使用样例
        if dict==None and txt==None:
            match.dict_add('abcde', '12345')
            match.dict_add('abc', '123')
            match.dict_add('cde', '345')
            match.dict_add('bcd', '234')
            dict={}
            txt = "hello!abcde!abc!"
        
        #建立被替换词->替换词词典
        for k in dict:
            self.dict_add(k, dict[k])

        #对当前字典建立AC树(拥有词典后建立一次即可，无需重复建立)
        self.createAcTree()


        print(f"txt:{txt}")
        #对txt进行匹配解析为三元组(默认使用最大优先匹配)
        self.do_match(txt,__SELECT_MODE__=0)

        #选择一种选择策略生成新的三元组(该方法已弃用)
        #self.do_select_triples(txt)

        #对三元组空白位置添加None (使不使用均可)
        #self.newParsedTriples=self.do_complete(txt,self.newParsedTriples)

        #对txt进行替换 使用的是新的三元组
        self.do_filter(txt)


    #添加一组被替换词和替换词
    def dict_add(self, keyword, val='\x00', strip=True,keyword_lower=False):
        str_repl=self.str_repl
        if strip:
            keyword = keyword.strip()  # 关键词丢弃首尾空白
        if keyword_lower:
            keyword = keyword.lower()  # 关键词变小写
        
        str_repl[keyword]=val
    
    #在已经添加完所有被替换词后创建AC自动机
    def createAcTree(self):
        ac=self.ac
        str_repl=self.str_repl
        ac.create(str_repl.keys())

    #对message进行匹配解析为三元组
    #__SELECT_MODE__ 解析时匹配的方式

    #__SELECT_MODE__={
    #   0 -> 前向最大化优先匹配(max_match)
    #   1 -> 全匹配(is_all)
    # }
    def do_match(self, message,__SELECT_MODE__=0):
        ac=self.ac
        rst = [(-1,-1,None)]
        def cb(b, e, v):
            #   0 -> 前向最大化优先匹配(max_match)
            if __SELECT_MODE__==0:
                keepCur=True
                while b<rst[-1][1]:
                    if b>rst[-1][0]:
                        keepCur=False
                        break
                    else:#b<=rst[-1][0]:
                        rst.pop()
                if keepCur:
                    rst.append((b, e, v))
                self.parsedTriples.append((b, e, v))
                
            #   1 -> 全匹配(is_all)
            elif __SELECT_MODE__==1:
                rst.append((b, e, v))
        ac.parseText(cb,message)
        del rst[0]
        #self.parsedTriples=rst
        self.newParsedTriples=rst
        print(f"do_match parsedTriples {self.parsedTriples}")
        print(f"do_match newParsedTriples {self.newParsedTriples}")
    

    #对匹配好的所有三元组进行选择
    #__SELECT_MODE__={
    #   0 -> 前向最大化优先匹配(max_match)
    #   1 -> ??
    # }

    #此方法已放弃，但可重新启用
    def do_select_triples(self,message,__SELECT_MODE__=0):
        #前向最大化优先匹配(max_match)
        #从前往后找最大长度的匹配
        rst=[]
        if __SELECT_MODE__==0:
            arr=sorted(self.parsedTriples,key=lambda x:(x[0],-x[1]))

            l=0
            while l<len(arr):
                end=arr[l][1]
                rst.append(arr[l])
                l+=1
                while l<len(arr) and arr[l][0]<end:
                    l+=1
            
            
        
        #补全三元组
        self.newParsedTriples=self.do_complete(message,rst)
        
        print(f"do_select_triples {self.newParsedTriples}")

    #将三元组没有替换的区域 添加为(b,e,None)
    #返回新的三元组
    def do_complete(self,message,triples):
        rst=[]
        if not triples:
            return
        
        if triples[0][0]>0:
            rst.append((0,triples[0][0],None))
        
        for i in range(len(triples)-1):
            rst.append(triples[i])
            if triples[i][1]<triples[i+1][0]:
                rst.append((triples[i][1],triples[i+1][0],None))
        rst.append(triples[-1])
        if triples[-1][1]<len(message):
            rst.append((triples[-1][1],len(message),None))

        print(f"do_complete {rst}")
        return rst

    #对message进行替换
    #是否do_complete都可转换
    def do_filter(self, message, repl="*"):
        rst=""
        npt=self.newParsedTriples

        pre=0
        for i in range(len(npt)):
            b,e,raw=npt[i]
            if pre<b:
                rst+=message[pre:b]
            if raw==None:
                rst+=message[b:e]
            else:
                rst+=self.str_repl[raw]
            pre=e
        rst+=message[pre:]

        print(f"do_filter {rst}")
        return rst
        


if __name__ == '__main__':
    #testcase1
    match = ac_match_t()
    match.dict_add('abcde', '12345')
    match.dict_add('abc', '123')
    match.dict_add('cde', '345')
    match.dict_add('bcd', '234')

    match.createAcTree()
    txt = "hello!abcde!abc!"
    print(f"txt:{txt}")
    #对message进行匹配解析为三元组
    match.do_match(txt)

    #选择一种选择策略生成新的三元组
    #match.do_select_triples(txt)

    #对三元组空白位置添加None (使不使用均可)
    #match.newParsedTriples=match.do_complete(txt,match.newParsedTriples)

    #对新的三元组中的词进行替换
    match.do_filter(txt)
    

    #testcase2
    #solve中集成了上面的方法
    # match = ac_match_t()
    # match.solve(dict=None,txt=None)


    
