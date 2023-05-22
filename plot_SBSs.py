import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
from matplotlib.pyplot import MultipleLocator
font = {'family' : 'SimSun',
    'weight' : 'bold',
    'size'  : '12'}
plt.rc('font', **font)        # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False) # 步骤二（解决坐标轴负数的负号显示问题）

# expericement 1
agent1_hit_radio_11 = [9.68967, 15.92624, 23.17725, 30.03348, 34.18620]
agent1_reward_11 = [189261.00000, 311313.00000, 448662.00000, 579990.00000, 657159.00000]
agent1_cost_11 = [1811839.00000, 1689787.00000, 1552438.00000, 1421110.00000, 1343941.00000]

agent1_hit_radio_21 = [14.09725, 22.42267, 34.62096, 44.87032, 51.16686]
agent2_hit_radio_21 = [13.66748, 21.96792, 34.08625, 44.75538, 50.88201]
hit_radio_21 = []
for i in range(len(agent1_hit_radio_11)):
    hit_radio_21.append(1/2*(agent1_hit_radio_21[i]+agent2_hit_radio_21[i]))

agent1_reward_21 = [246671.00000, 397662.00000, 609330.00000, 776976.00000, 898505.00000]
agent2_reward_21 = [241092.00000, 390151.00000, 598186.00000, 781666.00000, 883250.00000]
reward_21 = []
for i in range(len(agent1_hit_radio_11)):
    reward_21.append(1/2*(agent1_reward_21[i]+agent2_reward_21[i]))

agent1_cost_21 = [1754429.00000, 1603438.00000, 1391770.00000, 1224124.00000, 1102595.00000]
agent2_cost_21 = [1760008.00000, 1610949.00000, 1402914.00000, 1219434.00000, 1117850.00000]
cost_21 = []
for i in range(len(agent1_hit_radio_11)):
    cost_21.append(1/2*(agent1_cost_21[i]+agent2_cost_21[i]))

agent1_hit_radio_31 = [12.97786, 22.64255, 35.29559, 45.05022, 50.99195]
agent2_hit_radio_31 = [15.06172, 26.84524, 38.83364, 47.75873, 56.72380]
agent3_hit_radio_31 = [13.11279, 22.29274, 34.13623, 42.53660, 51.64659]
hit_radio_31 = []
for i in range(len(agent1_hit_radio_11)):
    hit_radio_31.append(1/3*(agent1_hit_radio_31[i]+agent2_hit_radio_31[i]+agent3_hit_radio_31[i]))

agent1_reward_31 = [227778.00000, 397230.00000, 617214.00000, 783222.00000, 885465.00000]
agent2_reward_31 = [254954.00000, 448363.00000, 668562.00000, 815697.00000, 966879.00000]
agent3_reward_31 = [231634.00000, 396071.00000, 594807.00000, 743756.00000, 897618.00000]
reward_31 = []
for i in range(len(agent1_hit_radio_11)):
    reward_31.append(1/3*(agent1_reward_31[i]+agent2_reward_31[i]+agent3_reward_31[i]))

agent1_cost_31 = [1773322.00000, 1603870.00000, 1383886.00000, 1217878.00000, 1115635.00000]
agent2_cost_31 = [1746146.00000, 1552737.00000, 1332538.00000, 1185403.00000, 1034221.00000]
agent3_cost_31 = [1769466.00000, 1605029.00000, 1406293.00000, 1257344.00000, 1103482.00000]
cost_31 = []
for i in range(len(agent1_hit_radio_11)):
    cost_31.append(1/3*(agent1_cost_31[i]+agent2_cost_31[i]+agent3_cost_31[i]))

agent1_hit_radio_41 = [13.84239, 23.70196, 35.72535, 43.65599, 51.16686]
agent2_hit_radio_41 = [15.68138, 26.52541, 40.60267, 47.17905, 52.36620]
agent3_hit_radio_41 = [15.57643, 26.07066, 37.82919, 46.70931, 54.68492]
agent4_hit_radio_41 = [13.06781, 23.25721, 34.35111, 44.62046, 50.90700]
hit_radio_41 = []
for i in range(len(agent1_hit_radio_11)):
    hit_radio_41.append(1/4*(agent1_hit_radio_41[i]+agent2_hit_radio_41[i]+agent3_hit_radio_41[i]+agent4_hit_radio_41[i]))

agent1_reward_41 = [239782.00000, 416888.00000, 619770.00000, 760928.00000, 893140.00000]
agent2_reward_41 = [266940.00000, 451460.00000, 689717.00000, 810091.00000, 910220.00000]
agent3_reward_41 = [258036.00000, 445550.00000, 651328.00000, 807203.00000, 938657.00000]
agent4_reward_41 = [231213.00000, 410996.00000, 598916.00000, 777166.00000, 878806.00000]
reward_41 = []
for i in range(len(agent1_hit_radio_11)):
    reward_41.append(1/4*(agent1_reward_41[i]+agent2_reward_41[i]+agent3_reward_41[i]++agent4_reward_41[i]))

agent1_cost_41 = [1761318.00000, 1584212.00000, 1381330.00000, 1240172.00000, 1107960.00000]
agent2_cost_41 = [1734160.00000, 1549640.00000, 1311383.00000, 1191009.00000, 1090880.00000]
agent3_cost_41 = [1743064.00000, 1555550.00000, 1349772.00000, 1193897.00000, 1062443.00000]
agent4_cost_41 = [1769887.00000, 1590104.00000, 1402184.00000, 1223934.00000, 1122294.00000]
cost_41 = []
for i in range(len(agent1_hit_radio_11)):
    cost_41.append(1/4*(agent1_cost_41[i]+agent2_cost_41[i]+agent3_cost_41[i]+agent4_cost_41[i]))

# expericement 2
agent1_hit_radio_12 = [9.70966, 15.24661, 23.74694, 28.51432, 33.19174]
agent1_reward_12 = [189557.00000, 296349.00000, 460848.00000, 550794.00000, 637458.00000]
agent1_cost_12 = [1811543.00000, 1704751.00000, 1540252.00000, 1450306.00000, 1363642.00000]

agent1_hit_radio_22 = [13.70246, 22.47264, 35.88526, 44.15072, 50.65214]
agent2_hit_radio_22 = [13.81740, 22.11284, 35.92024, 44.51052, 50.47724]
hit_radio_22 = []
for i in range(len(agent1_hit_radio_12)):
    hit_radio_22.append(1/2*(agent1_hit_radio_22[i]+agent2_hit_radio_22[i]))

agent1_reward_22 = [242062.00000, 402219.00000, 618139.00000, 766403.00000, 887275.00000]
agent2_reward_22 = [243289.00000, 394260.00000, 630620.00000, 779672.00000, 882779.00000]
reward_22 = []
for i in range(len(agent1_hit_radio_11)):
    reward_22.append(1/2*(agent1_reward_22[i]+agent2_reward_22[i]))

agent1_cost_22 = [1759038.00000, 1598881.00000, 1382961.00000, 1234697.00000, 1113825.00000]
agent2_cost_22 = [1757811.00000, 1606840.00000, 1370480.00000, 1221428.00000, 1118321.00000]
cost_22 = []
for i in range(len(agent1_hit_radio_11)):
    cost_22.append(1/2*(agent1_cost_22[i]+agent2_cost_22[i]))

agent1_hit_radio_32 = [14.67193, 23.34216, 34.87582, 44.57049, 50.57219]
agent2_hit_radio_32 = [17.07061, 24.98626, 38.97356, 46.54440, 55.62441]
agent3_hit_radio_32 = [14.45205, 23.21723, 34.91080, 42.53161, 50.41727]
hit_radio_32 = []
for i in range(len(agent1_hit_radio_11)):
    hit_radio_32.append(1/3*(agent1_hit_radio_32[i]+agent2_hit_radio_32[i]+agent3_hit_radio_32[i]))

agent1_reward_32 = [256712.00000, 412875.00000, 611480.00000, 782112.00000, 884631.00000]
agent2_reward_32 = [283735.00000, 429229.00000, 672301.00000, 800520.00000, 949816.00000]
agent3_reward_32 = [255391.00000, 405171.00000, 607447.00000, 738264.00000, 878713.00000]
reward_32 = []
for i in range(len(agent1_hit_radio_11)):
    reward_32.append(1/3*(agent1_reward_32[i]+agent2_reward_32[i]+agent3_reward_32[i]))

agent1_cost_32 = [1744388.00000, 1588225.00000, 1389620.00000, 1218988.00000, 1116469.00000]
agent2_cost_32 = [1717365.00000, 1571871.00000, 1328799.00000, 1200580.00000, 1051284.00000]
agent3_cost_32 = [1745709.00000, 1595929.00000, 1393653.00000, 1262836.00000, 1122387.00000]
cost_32 = []
for i in range(len(agent1_hit_radio_11)):
    cost_32.append(1/3*(agent1_cost_32[i]+agent2_cost_32[i]+agent3_cost_32[i]))

agent1_hit_radio_42 = [13.84239, 23.70196, 35.72535, 43.65599, 51.16686]
agent2_hit_radio_42 = [15.68138, 26.52541, 40.60267, 47.17905, 52.36620]
agent3_hit_radio_42 = [15.57643, 26.07066, 37.82919, 46.70931, 54.68492]
agent4_hit_radio_42 = [13.06781, 23.25721, 34.35111, 44.62046, 50.90700]
hit_radio_42 = []
for i in range(len(agent1_hit_radio_11)):
    hit_radio_42.append(1/4*(agent1_hit_radio_42[i]+agent2_hit_radio_42[i]+agent3_hit_radio_42[i]+agent4_hit_radio_42[i]))

agent1_reward_42 = [239782.00000, 416888.00000, 619770.00000, 760928.00000, 893140.00000]
agent2_reward_42 = [266940.00000, 451460.00000, 689717.00000, 810091.00000, 910220.00000]
agent3_reward_42 = [258036.00000, 445550.00000, 651328.00000, 807203.00000, 938657.00000]
agent4_reward_42 = [231213.00000, 410996.00000, 598916.00000, 777166.00000, 878806.00000]
reward_42 = []
for i in range(len(agent1_hit_radio_11)):
    reward_42.append(1/4*(agent1_reward_42[i]+agent2_reward_42[i]+agent3_reward_42[i]++agent4_reward_42[i]))

agent1_cost_42 = [1761318.00000, 1584212.00000, 1381330.00000, 1240172.00000, 1107960.00000]
agent2_cost_42 = [1734160.00000, 1549640.00000, 1311383.00000, 1191009.00000, 1090880.00000]
agent3_cost_42 = [1743064.00000, 1555550.00000, 1349772.00000, 1193897.00000, 1062443.00000]
agent4_cost_42 = [1769887.00000, 1590104.00000, 1402184.00000, 1223934.00000, 1122294.00000]
cost_42 = []
for i in range(len(agent1_hit_radio_11)):
    cost_42.append(1/4*(agent1_cost_42[i]+agent2_cost_42[i]+agent3_cost_42[i]+agent4_cost_42[i]))

# expericement 3
agent1_hit_radio_13 = [9.69467, 16.06117, 23.74694, 28.51432, 33.19174]
agent1_reward_13 = [189660.00000, 313286.00000, 460848.00000, 550794.00000, 637458.00000]
agent1_cost_13 = [1811440.00000, 1687814.00000, 1540252.00000, 1450306.00000, 1363642.00000]

agent1_hit_radio_23 = [13.69247, 23.64200, 35.14567, 44.71041, 51.12188]
agent2_hit_radio_23 = [13.68747, 23.52206, 34.69092, 45.30008, 50.89201]
hit_radio_23 = []
for i in range(len(agent1_hit_radio_12)):
    hit_radio_23.append(1/2*(agent1_hit_radio_23[i]+agent2_hit_radio_23[i]))

agent1_reward_23 = [242096.00000, 418677.00000, 612003.00000, 777844.00000, 890825.00000]
agent2_reward_23 = [241736.00000, 410355.00000, 604226.00000, 792744.00000, 883502.00000]
reward_23 = []
for i in range(len(agent1_hit_radio_11)):
    reward_23.append(1/2*(agent1_reward_23[i]+agent2_reward_23[i]))

agent1_cost_23 = [1759004.00000, 1582423.00000, 1389097.00000, 1223256.00000, 1110275.00000]
agent2_cost_23 = [1759364.00000, 1590745.00000, 1396874.00000, 1208356.00000, 1117598.00000]
cost_23 = []
for i in range(len(agent1_hit_radio_11)):
    cost_23.append(1/2*(agent1_cost_23[i]+agent2_cost_23[i]))

agent1_hit_radio_33 = [14.62196, 22.85243, 35.17565, 44.46055, 50.72210]
agent2_hit_radio_33 = [16.44595, 24.09175, 40.04298, 49.50777, 54.28015]
agent3_hit_radio_33 = [13.38764, 22.82744, 35.18565, 44.41557, 50.10244]
hit_radio_33 = []
for i in range(len(agent1_hit_radio_11)):
    hit_radio_33.append(1/3*(agent1_hit_radio_33[i]+agent2_hit_radio_33[i]+agent3_hit_radio_33[i]))

agent1_reward_33 = [258577.00000, 405841.00000, 610836.00000, 773181.00000, 890490.00000]
agent2_reward_33 = [277405.00000, 413637.00000, 682608.00000, 844739.00000, 931483.00000]
agent3_reward_33 = [238281.00000, 404540.00000, 608972.00000, 765574.0000, 870227.00000]
reward_33 = []
for i in range(len(agent1_hit_radio_11)):
    reward_33.append(1/3*(agent1_reward_33[i]+agent2_reward_33[i]+agent3_reward_33[i]))

agent1_cost_33 = [1742523.00000, 1595259.00000, 1390264.00000, 1227919.00000, 1110610.00000]
agent2_cost_33 = [1723695.00000, 1587463.00000, 1318492.00000, 1156361.00000, 1069617.00000]
agent3_cost_33 = [1762819.00000, 1596560.00000, 1392128.00000, 1235526.00000, 1130873.00000]
cost_33 = []
for i in range(len(agent1_hit_radio_11)):
    cost_33.append(1/3*(agent1_cost_33[i]+agent2_cost_33[i]+agent3_cost_33[i]))

agent1_hit_radio_43 = [13.66249, 22.72750, 35.52546, 44.69542, 51.52666]
agent2_hit_radio_43 = [15.12668, 27.10009, 37.93414, 50.15741, 54.33012]
agent3_hit_radio_43 = [16.45095, 26.05067, 38.68872, 47.14907, 55.37954]
agent4_hit_radio_43 = [14.74189, 22.68752, 34.98076, 43.54605, 52.26625]
hit_radio_43 = []
for i in range(len(agent1_hit_radio_11)):
    hit_radio_43.append(1/4*(agent1_hit_radio_43[i]+agent2_hit_radio_43[i]+agent3_hit_radio_43[i]+agent4_hit_radio_43[i]))

agent1_reward_43 = [242762.00000, 394731.00000, 625306.00000, 783711.00000, 893025.00000]
agent2_reward_43 = [256174.00000, 454370.00000, 651567.00000, 810091.00000, 941018.00000]
agent3_reward_43 = [278500.00000, 439077.00000, 662195.00000, 807203.00000, 947636.00000]
agent4_reward_43 = [262509.00000, 405768.00000, 608680.00000, 759313.00000, 912671.00000]
reward_43 = []
for i in range(len(agent1_hit_radio_11)):
    reward_43.append(1/4*(agent1_reward_43[i]+agent2_reward_43[i]+agent3_reward_43[i]++agent4_reward_43[i]))

agent1_cost_43 = [1758338.00000, 1606369.00000, 1375794.00000, 1217389.00000, 1108075.00000]
agent2_cost_43 = [1744926.00000, 1546730.00000, 1349533.00000, 1147724.00000, 1060082.00000]
agent3_cost_43 = [1722600.00000, 1562023.00000, 1338905.00000, 1193714.00000, 1053464.00000]
agent4_cost_43 = [1738591.00000, 1595332.00000, 1392420.00000, 1241787.00000, 1088429.00000]
cost_43 = []
for i in range(len(agent1_hit_radio_11)):
    cost_43.append(1/4*(agent1_cost_43[i]+agent2_cost_43[i]+agent3_cost_43[i]+agent4_cost_43[i]))

hit_radio_4 = []
for i in range(len(agent1_hit_radio_11)):
    hit_radio_4.append(1/3*(hit_radio_41[i]+hit_radio_42[i]+hit_radio_43[i]))
hit_radio_3 = []
for i in range(len(agent1_hit_radio_11)):
    hit_radio_3.append(1/3*(hit_radio_31[i]+hit_radio_32[i]+hit_radio_33[i]))
hit_radio_2 = []
for i in range(len(agent1_hit_radio_11)):
    hit_radio_2.append(1/3*(hit_radio_21[i]+hit_radio_22[i]+hit_radio_23[i]))
hit_radio_1 = []
for i in range(len(agent1_hit_radio_11)):
    hit_radio_1.append(1/3*(agent1_hit_radio_11[i]+agent1_hit_radio_12[i]+agent1_hit_radio_13[i]))
reward_3 = []
for i in range(len(agent1_hit_radio_11)):
    reward_3.append(1/3*(reward_31[i]+reward_32[i]+reward_33[i]))
reward_2 = []
for i in range(len(agent1_hit_radio_11)):
    reward_2.append(1/3*(reward_21[i]+reward_22[i]+reward_23[i]))
reward_4 = []
for i in range(len(agent1_hit_radio_11)):
    reward_4.append(1/3*(reward_41[i]+reward_42[i]+reward_43[i]))
reward_1 = []
for i in range(len(agent1_hit_radio_11)):
    reward_1.append(1/3*(agent1_reward_11[i]+agent1_reward_12[i]+agent1_reward_13[i]))
cost_3 = []
for i in range(len(agent1_hit_radio_11)):
    cost_3.append(1/3*(cost_31[i]+cost_32[i]+cost_33[i]))
cost_2 = []
for i in range(len(agent1_hit_radio_11)):
    cost_2.append(1/3*(cost_21[i]+cost_22[i]+cost_23[i]))
cost_4 = []
for i in range(len(agent1_hit_radio_11)):
    cost_4.append(1/3*(cost_41[i]+cost_42[i]+cost_43[i]))
cost_1 = []
for i in range(len(agent1_hit_radio_11)):
    cost_1.append(1/3*(agent1_cost_11[i]+agent1_cost_12[i]+agent1_cost_13[i]))

import numpy as np
plt.xlabel('缓存容量',fontdict={"family": "SimSun", "size": 15})
plt.ylabel('缓存命中率',fontdict={"family": "SimSun", "size": 15})
name_list = ['50', '100', '200', '300', '400']
width = 0.21
ax = plt.gca()
ax.yaxis.set_minor_locator(plt.MultipleLocator(200000))
ax.grid(which="both",linestyle='--',zorder=0)
#plt.ylim(1500000,2100000)
plt.bar(np.arange(5)-width, hit_radio_1, width=width, label='1 SBS',color='#575C73',zorder=10)
plt.bar(np.arange(5)+0.01, hit_radio_2, width=width, label='2 SBSs',color='#778BBE',zorder=10)
plt.bar(np.arange(5)+width+0.02, hit_radio_3, width=width, label='3 SBSs',color='#91B2D7',zorder=10)
plt.bar(np.arange(5)+width*2+0.03, hit_radio_4, width=width, label='4 SBSs',color='#807C93',zorder=10)
plt.xticks(np.arange(5)+width/2,name_list)
plt.legend()
plt.show()

plt.xlabel('缓存容量',fontdict={"family": "SimSun", "size": 15})
plt.ylabel('成本',fontdict={"family": "SimSun", "size": 15})
name_list = ['50', '100', '200', '300', '400']
width = 0.21
ax = plt.gca()
ax.yaxis.set_minor_locator(plt.MultipleLocator(200000))
ax.grid(which="both",linestyle='--',zorder=0)
plt.ylim(1000000,1850000)
plt.bar(np.arange(5)-width, cost_1, width=width, label='1 SBS',color='#CADE72',zorder=10)
plt.bar(np.arange(5)+0.01, cost_2, width=width, label='2 SBSs',color='#92C15D',zorder=10)
plt.bar(np.arange(5)+width+0.02, cost_3, width=width, label='3 SBSs',color='#BDDCBE',zorder=10)
plt.bar(np.arange(5)+width*2+0.03, cost_4, width=width, label='4 SBSs',color='#81C196',zorder=10)
plt.xticks(np.arange(5)+width/2,name_list)
plt.legend()
plt.show()

plt.xlabel('缓存容量',fontdict={"family": "SimSun", "size": 15})
plt.ylabel('奖励',fontdict={"family": "SimSun", "size": 15})
name_list = ['50', '100', '200', '300', '400']
width = 0.21
ax = plt.gca()
ax.yaxis.set_minor_locator(plt.MultipleLocator(200000))
ax.grid(which="both",linestyle='--',zorder=0)
plt.ylim(160000,930000)
plt.bar(np.arange(5)-width, reward_1, width=width, label='1 SBS',color='#9FAC88',zorder=10)
plt.bar(np.arange(5)+0.01, reward_2, width=width, label='2 SBSs',color='#E0A97C',zorder=10)
plt.bar(np.arange(5)+width+0.02, reward_3, width=width, label='3 SBSs',color='#E7E786',zorder=10)
plt.bar(np.arange(5)+width*2+0.03, reward_4, width=width, label='4 SBSs',color='#C4D0CC',zorder=10)
plt.xticks(np.arange(5)+width/2,name_list)
plt.legend()
plt.show()
