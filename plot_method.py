import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
from matplotlib.pyplot import MultipleLocator

TS1_cost_2 = [1955193, 1860657, 1639421, 1396549, 1218604]
TS2_cost_2 = [1952723, 1854609, 1623259, 1400834, 1216641]

TS1_hit_radio_2 = [3.013342661536155, 8.820148918095047, 22.16780770576183, 35.950227374943786, 45.72485133176753]
TS2_hit_radio_2 = [3.093298685722852, 9.219929039028536, 22.567587826695316, 35.71035930238369, 46.31452701014442]

greedy1_cost_2 = [1808433.4000000001, 1658442.1, 1443666.0, 1273731.0000000002, 1127604.5]
greedy2_cost_2 = [1821925.2, 1681531.9000000001, 1484361.5, 1321113.3, 1176830.2]

greedy1_hit_radio_2 = [9.786617360451753, 17.42041876967668, 28.38588776173105, 37.06761281295288, 44.54350107440907]
greedy2_hit_radio_2 = [9.107490880015991, 16.24306631352756, 26.320023986807257, 34.7029133976313, 42.09734645944731]

random1_cost_2 = [1960681, 1921833, 1846713, 1758291, 1688768]
random2_cost_2 = [1964622, 1922095, 1840244, 1757211, 1688302]

random1_hit_radio_2 = [2.653540552696017, 5.282094847833691, 10.274349107990604, 15.776323022337715, 20.283843885862776]
random2_hit_radio_2 = [2.4736394982759484, 5.142171805506971, 10.484233671480686, 16.10114437059617, 20.6936185098196]

TS_cost_2 = []
for i in range(len(TS1_cost_2)):
    TS_cost_2.append(1/2*(TS1_cost_2[i]+TS2_cost_2[i]))
TS_hit_radio_2 = []
for i in range(len(TS1_cost_2)):
    TS_hit_radio_2.append(1/2*(TS1_hit_radio_2[i]+TS2_hit_radio_2[i]))
greedy_cost_2 = []
for i in range(len(TS1_cost_2)):
    greedy_cost_2.append(1/2*(greedy1_cost_2[i]+greedy2_cost_2[i]))
greedy_hit_radio_2 = []
for i in range(len(TS1_cost_2)):
    greedy_hit_radio_2.append(1/2*(greedy1_hit_radio_2[i]+greedy2_hit_radio_2[i]))
random_cost_2 = []
for i in range(len(TS1_cost_2)):
    random_cost_2.append((1/2*(random1_cost_2[i]+random2_cost_2[i])))
random_hit_radio_2 = []
for i in range(len(TS1_cost_2)):
    random_hit_radio_2.append(1/2*(random1_hit_radio_2[i]+random2_hit_radio_2[i]))

#proposed
agent1_hit_radio_21 = [14.09725, 22.42267, 34.62096, 44.87032, 51.16686]
agent2_hit_radio_21 = [13.66748, 21.96792, 34.08625, 44.75538, 50.88201]
hit_radio_21 = []
for i in range(len(agent1_hit_radio_21)):
    hit_radio_21.append(1/2*(agent1_hit_radio_21[i]+agent2_hit_radio_21[i]))

agent1_cost_21 = [1754429.00000, 1603438.00000, 1391770.00000, 1224124.00000, 1102595.00000]
agent2_cost_21 = [1760008.00000, 1610949.00000, 1402914.00000, 1219434.00000, 1117850.00000]
cost_21 = []
for i in range(len(agent1_hit_radio_21)):
    cost_21.append(1/2*(agent1_cost_21[i]+agent2_cost_21[i]))

agent1_hit_radio_22 = [13.70246, 22.47264, 35.88526, 44.15072, 50.65214]
agent2_hit_radio_22 = [13.81740, 22.11284, 35.92024, 44.51052, 50.47724]
hit_radio_22 = []
for i in range(len(agent1_hit_radio_22)):
    hit_radio_22.append(1/2*(agent1_hit_radio_22[i]+agent2_hit_radio_22[i]))

agent1_cost_22 = [1759038.00000, 1598881.00000, 1382961.00000, 1234697.00000, 1113825.00000]
agent2_cost_22 = [1757811.00000, 1606840.00000, 1370480.00000, 1221428.00000, 1118321.00000]
cost_22 = []
for i in range(len(agent1_hit_radio_21)):
    cost_22.append(1/2*(agent1_cost_22[i]+agent2_cost_22[i]))

agent1_hit_radio_23 = [13.69247, 23.64200, 35.14567, 44.71041, 51.12188]
agent2_hit_radio_23 = [13.68747, 23.52206, 34.69092, 45.30008, 50.89201]
hit_radio_23 = []
for i in range(len(agent1_hit_radio_22)):
    hit_radio_23.append(1/2*(agent1_hit_radio_23[i]+agent2_hit_radio_23[i]))

agent1_cost_23 = [1759004.00000, 1582423.00000, 1389097.00000, 1223256.00000, 1110275.00000]
agent2_cost_23 = [1759364.00000, 1590745.00000, 1396874.00000, 1208356.00000, 1117598.00000]
cost_23 = []
for i in range(len(agent1_hit_radio_21)):
    cost_23.append(1/2*(agent1_cost_23[i]+agent2_cost_23[i]))

hit_radio_2 = []
for i in range(len(agent1_hit_radio_22)):
    hit_radio_2.append(1/3*(hit_radio_21[i]+hit_radio_22[i]+hit_radio_23[i]))

cost_2 = []
for i in range(len(agent1_hit_radio_21)):
    cost_2.append(1/3*(cost_23[i]+cost_22[i]+cost_21[i]))


import numpy as np

plt.xlabel('缓存容量',fontdict={"family": "SimSun", "size": 15})
plt.ylabel('成本',fontdict={"family": "SimSun", "size": 15})
name_list = ['50', '100', '200', '300', '400']
width = 0.22
ax = plt.gca()
ax.yaxis.set_minor_locator(plt.MultipleLocator(200000))
ax.grid(which="both",linestyle='--',zorder=0)
plt.ylim(1000000,2100000)
plt.bar(np.arange(5)-width/2*3, cost_2, width=width, label='CMREF',color='#AB9B8C',zorder=10)
plt.bar(np.arange(5)-width/2+0.01, greedy_cost_2, width=width, label='C-ε-greedy',color='#DEA675',zorder=10)
plt.bar(np.arange(5)+width/2+0.02, TS_cost_2, width=width, label='Thompson sampling',color='#977250',zorder=10)
plt.bar(np.arange(5)+3*width/2+0.03, random_cost_2, width=width, label='Random',color='#674e38',zorder=10)
plt.xticks(np.arange(5),name_list)
plt.legend()
plt.show()

plt.xlabel('缓存容量',fontdict={"family": "SimSun", "size": 15})
plt.ylabel('缓存命中率',fontdict={"family": "SimSun", "size": 15})
name_list = ['50', '100', '200', '300', '400']
width = 0.22
ax = plt.gca()
ax.yaxis.set_minor_locator(plt.MultipleLocator(200000))
ax.grid(which="both",linestyle='--',zorder=0)
#plt.ylim(1000000,2100000)
plt.bar(np.arange(5)-width/2*3, random_hit_radio_2, width=width, label='Random',color='#AB9B8C',zorder=10)
plt.bar(np.arange(5)-width/2+0.01, TS_hit_radio_2, width=width, label='Thompson sampling',color='#DEA675',zorder=10)
plt.bar(np.arange(5)+width/2+0.02, greedy_hit_radio_2, width=width, label='N-ε-greedy',color='#977250',zorder=10)
plt.bar(np.arange(5)+3*width/2+0.03, hit_radio_2, width=width, label='CMREF',color='#674e38',zorder=10)
plt.xticks(np.arange(5),name_list)
plt.legend(prop={'family' : 'Times New Roman', 'size' : 10})
plt.show()