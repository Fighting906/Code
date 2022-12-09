#AC自动机 原版
#根据words创建ac树
#根据输入的txt进行多模式匹配

from collections import deque
from typing import List

#Trie树Node节点
class TrieNode:
    def __init__(self) -> None:
        self.letter=None #当前节点字母
        self.next={} #当前节点的下一节点 {str:TrieNode}
        self.isEnd=False #当前节点是否是结束节点
        
        self.word=None #当前节点所表示的前缀单词
        self.information=[] #当前节点结束所包含的单词

        self.fail=None #当前节点所指向的fail指针
        self.pre=None #当前节点的父节点


#AC自动机多模式匹配
class Aho_Corasick_automaton:
    def __init__(self) -> None:
        self.root=TrieNode()

    #对words创建Trie树和Fail指针
    def create(self,words: List[str]):
        self.root=TrieNode()
        self.__createTrieTree(words)
        self.__createFail()

    #根据当前words创建Trie树
    def __createTrieTree(self,words: List[str]):
        root=self.root
        for word in words:
            node=root
            prenode=root
            cur=''
            for l in word:
                cur+=l
                if l not in node.next:
                    node.next[l]=TrieNode()
                node=node.next[l]
                node.letter=l
                node.word=cur
                node.pre=prenode
                prenode=node

            node.isEnd=True
            node.information.append(cur)
            

    #在已有Trie树的情况下 给节点添加Fail指针
    def __createFail(self):
        root=self.root
        q=deque([root])
        #层序遍历为Trie树节点增加Fail指针
        while q:
            for i in range(len(q)):
                node=q.popleft()
                for l in node.next:
                    q.append(node.next[l])

                #根节点fail指针为None 无需添加fail指针
                if node==root:
                    continue
                    
                #fafail为父亲的fail指针
                fafail=node.pre.fail
                
                #核心思想为 直到找到fafail含有字母letter 否则就继续寻找fafail的fail节点
                while fafail and node.letter not in fafail.next:
                    fafail=fafail.fail
                
                #fafail为空的情况下 代表已经递归到root节点
                #root不包含node.letter
                if not fafail:
                    node.fail=root

                #node.letter in fafail
                else:
                    node.fail=fafail.next[node.letter]
                
                #如果node的fail节点为终止节点 那么将其信息加入到当前节点
                if node.fail.isEnd:
                    node.information+=node.fail.information

    #将一个字符串word 按照已有的ac自动机解析
    def parseText(self,word):
        root=self.root
        node=root
        ret=[]
        for l in word:
            #匹配不上就去fail
            while node and l not in node.next:
                node=node.fail
            
            #root也没有l
            if not node:
                node=root
                continue
            #node中有l
            else:
                node=node.next[l]
                #如果node有information
                if node.isEnd:
                    ret+=node.information
        return ret


            
if __name__=="__main__":
    ac=Aho_Corasick_automaton()
    #测试样例1
    #对应图在下面链接
    #https://blog.csdn.net/kilig_CSM/article/details/119341634?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-119341634-blog-120368878.pc_relevant_landingrelevant&spm=1001.2101.3001.4242.2&utm_relevant_index=4
    words1=['he','she','hers','his']
    parsetext11='ahishers'
    parsetext21='ushers'
    ac.create(words1)
    print(ac.parseText(parsetext11))#{his,she,he,hers}
    print(ac.parseText(parsetext21))#{she,he,hers}

    #测试样例2
    words2=['a','aa','aaa','ba']
    parsetext21='aaabaa'
    ac.create(words2)
    print(ac.parseText(parsetext21))#{a,aa,a,aaa,aa,a,ba,a,aa,a}




    