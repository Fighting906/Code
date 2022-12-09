#Log 对(b,e,word)的匹配进行持续记录

class pos_t:
    """匹配位置信息记录"""

    def __init__(self, begin, end, txt, txtl=None, txtr=None):
        self.begin = begin  # 开始位置[
        self.end = end  # 结束位置)
        self.txt = txt  # 开始结束对应的文本
        self.txt_left = txtl  # 开始位置左侧的文本
        self.txt_right = txtr  # 结束位置右侧的文本


class rep_t:
    """文本替换信息记录"""

    def __init__(self):
        self.src = None  # 匹配的源信息
        self.dst = None  # 替换后的信息
        self.tag = None  # 可选的标记

    def rec_src(self, begin, end, txt, txtl='', txtr=''):
        self.src = pos_t(begin, end, txt, txtl, txtr)

    def rec_dst(self, begin, end, txt):
        self.dst = pos_t(begin, end, txt)

    def __repr__(self):
        if self.src is None or self.dst is None:
            return 'EMPTY'
        return f"[{self.src.begin}:{self.src.end}='{self.src.txt}']=>[{self.dst.begin}:{self.dst.end}='{self.dst.txt}']"


class rep_rec_t:
    """文本替换信息的持续记录器"""

    def __init__(self, txt=None, rst=None, lr_len=3):
        self.adj = 0  # 目标偏移增量
        self.txt = txt  # 原始文本串
        self.lr_len = lr_len  # 额外记录的原始串左右内容长度
        self.rst = rst

    def make(self, begin, end, dst_txt, tag=None):
        """生成替换记录对象.将[begin,end)的内容替换为dst_txt"""
        if begin > end:
            return None
        ri = rep_t()
        ri.tag = tag

        # 匹配位置的左右内容
        txtl = self.txt[0:begin] if begin <= self.lr_len else self.txt[begin:begin + self.lr_len]
        txtr = self.txt[end: end + self.lr_len]
        # 记录匹配信息
        ri.rec_src(begin, end, self.txt[begin:end], txtl, txtr)

        # 计算替换后的位置信息,需要持续考虑"替换目标的长度增量"
        dst_begin = begin + self.adj
        dst_end = dst_begin + len(dst_txt)
        ri.rec_dst(dst_begin, dst_end, dst_txt)

        # 迭代更新结果文本中的长度增量
        self.adj += len(dst_txt) - (end - begin)
        if self.rst is not None:
            if len(self.rst) > 0:
                assert (ri.src.begin >= self.rst[-1].src.end)
                assert (ri.dst.begin >= self.rst[-1].dst.end)
            self.rst.append(ri)
        return ri