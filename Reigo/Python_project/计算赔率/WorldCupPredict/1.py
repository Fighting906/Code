
import pandas as pd
import numpy as np


#赢 平 输 赔率
def solve(winodds,tieodds,loseodds):
    #总投入钱数
    all_input=100

    #赢 平 输 预期概率
    winprob=0.8
    tieprob=0.1
    loseprob=0.1

    #赢平输投入对应钱数 对应 赢平输时的利益
    ipt_profit={}

    data_need=np.zeros((1,6))
    for win_input in range(0,all_input+1,2):
        for lose_input in range(0,all_input-win_input+1,2):
            tie_input=all_input-win_input-lose_input
            
            #正数个数
            cnt=0
            #if Win profits
            win_profit=winodds*win_input-all_input
            #if Tie Profits
            tie_profit=tieodds*tie_input-all_input
            #if Lose Profits
            lose_profit=loseodds*lose_input-all_input
            
            if win_profit>=0:
                cnt+=1
            if tie_profit>=0:
                cnt+=1
            if lose_profit>=0:
                cnt+=1
            
            if cnt<=1:
                continue


            ipt_profit[(win_input,tie_input,lose_input)]=(win_profit,tie_profit,lose_profit)
            #ws.write(win_input,tie_input,lose_input,win_profit,tie_profit,lose_profit)
            cur_row=np.array([[win_input,tie_input,lose_input,win_profit,tie_profit,lose_profit]])
            data_need=np.r_[data_need,cur_row]

    data_need_df=pd.DataFrame(data_need)
    data_need_df.to_csv("data.csv",index=False,header=['赢投入','平投入','输投入','赢利益','平利益','输利益'])
    return ipt_profit


#赢 平 输 赔率
solve(8.7,4.6,1.24)



