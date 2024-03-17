# read dir models and plot in one figure
import matplotlib.pyplot as plt
path1 = 'models/meibomian_gland2_ResUnetplusplus_woDS/log.csv'
path2 = 'models/meibomian_gland3_UNet_woDS/log.csv'
path3 = 'models/meibomian_gland4_AttU_Net_woDS/log.csv'
path4 = 'models/meibomian_gland_NestedUNet_woDS/log.csv'
path5 = 'models/meibomian_gland4_R2U_Net_woDS/log.csv'
path6 = 'models/meibomian_gland2_RP_Unet_woDS/log.csv'
path7 = 'models/meibomian_gland_R2AttU_Net_woDS/log.csv'
path8 = 'models/meibomian_gland4_Yolov8n_wDS/log.csv'
path9 = 'models/meibomian_gland_Mask-RCNN/log.csv'

loss1 = []
x = [i for i in range(1, 101)]
with open(path1, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            loss1.append(float(line.split(',')[2]))

loss2 = []
with open(path2, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            loss2.append(float(line.split(',')[2]))

loss3 = []
with open(path3, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            loss3.append(float(line.split(',')[2]))

loss4 = []
with open(path4, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            loss4.append(float(line.split(',')[2]))

loss5 = []
with open(path5, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            loss5.append(float(line.split(',')[2]))

loss6 = []
with open(path6, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            loss6.append(float(line.split(',')[2]))

loss7 = []
with open(path7, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            loss7.append(float(line.split(',')[2]))

loss8 = []
with open(path8, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            loss8.append(float(line.split(',')[2]))

loss9 = []
with open(path9, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            loss9.append(float(line.split(',')[2]))

plt.plot(x, loss1, 'o-', color='red', linewidth=1, label='Res-Unet++')
plt.plot(x, loss2, 'o-', color='cornflowerblue', linewidth=1, label='Unet')
plt.plot(x, loss3, 'o-', color='chartreuse', linewidth=1, label='Att-Unet')
plt.plot(x, loss4, 'o-', color='darkorange', linewidth=1, label='Unet++')
plt.plot(x, loss5, 'o-', color='saddlebrown', linewidth=1, label='R2-Unet')
plt.plot(x, loss6, '*-', color='purple', linewidth=1, label='RP-Unet')
plt.plot(x, loss7, 'o-', color='goldenrod', linewidth=1, label='R2Att-Unet')
plt.plot(x, loss8, 'o-', color='darkgrey', linewidth=1, label='YOLOv8n')
plt.plot(x, loss9, 'o-', color='aqua', linewidth=1, label='Mask-RCNN')

plt.xlabel('(c) Epoch', fontproperties = 'Times New Roman', size=18)
plt.ylabel('Loss', fontproperties = 'Times New Roman', size=18)
plt.legend(loc="best", frameon=False)  # 图例

plt.show()



