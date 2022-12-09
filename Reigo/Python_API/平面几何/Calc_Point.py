from math import sqrt

#已知一点，和一向量的方向和长度，计算令一点坐标
#Param：
# point 已知点坐标[x,y]
# vecDir 向量方向[a,b]
# vecL 向量长度L
# keepDir 是否与vecDir方向一致 
# keepDir=True为严格方向一致(返回一个根) keepDir=False为与vecDir平行即可(返回两个根)
#Return:
# [[x1,y1]] or [[x1,y1],[x2,y2]]
def CalculatePointToPoint(point,vecDir,vecL,keepDir=False):
    dx,dy=point
    a,b=vecDir
    h=vecL
    if a==0:
        x=dx
        y1=dy+h
        y2=dy-h
        if keepDir==False:
            return [[x,y1],[x,y2]]
        #y1方向一致
        if (y1-dy)*b>0:
            return [[x,y1]]
        #y2方向一致
        else:
            return [[x,y2]]
    else:
        bda=b/a
        temp=sqrt(h*h/(1+bda*bda))
        x1=dx+temp
        y1=dy+bda*temp

        x2=dx-temp
        y2=dy-bda*temp

        if keepDir==False:
            return [[x1,y1],[x2,y2]]
        
        #y1方向一致
        if (y1-dy)*b>0:
            return [[x1,y1]]
        #y2方向一致
        else:
            return [[x2,y2]]
    return [[None]]

#已知一向量的方向，计算其法向量
#vecDir 向量方向[a,b]
def CalculateNormalVector(vecDir):
    a,b=vecDir
    return [b,-a]

#已知两点，在两点连线的ratio处找到另一点，以该点做线段的垂直，长度为h，求令两点
#param:
# pointa,pointb [x,y] 点a和点b
# ratio 两点之间的比例
# h 法线长度
def solve(pointa,pointb,ratio,h):
    x1,y1=pointa
    x2,y2=pointb
    #[a,b] 两点的向量
    a,b=x2-x1,y2-y1
    x3=a*ratio+x1
    y3=b*ratio+y1
    
    #计算a，b的法向量
    af,bf=CalculateNormalVector([a,b])
    return CalculatePointToPoint([x3,y3],[af,bf],h,False)

if __name__=='__main__':
    print(CalculatePointToPoint([0,0],[0,1],7,False))

    #点a 点b ratio h
    print(solve([8,6],[0,0],0.7,4))
    print(solve([0,0],[8,6],0.3,4))
    print(solve([0,0],[500,0],0.6,0.2))