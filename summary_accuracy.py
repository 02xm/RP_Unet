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

acc1 = []
x = [i for i in range(1, 101)]
with open(path1, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            acc1.append(float(line.split(',')[7]))

acc2 = []
with open(path2, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            acc2.append(float(line.split(',')[7]))

acc3 = []
with open(path3, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            acc3.append(float(line.split(',')[7]))

acc4 = []
with open(path4, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            acc4.append(float(line.split(',')[7]))

acc5 = []
with open(path5, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            acc5.append(float(line.split(',')[7]))

acc6 = []
with open(path6, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            acc6.append(float(line.split(',')[7]))

acc7 = []
with open(path7, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            acc7.append(float(line.split(',')[7]))

acc8 = []
with open(path8, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            acc8.append(float(line.split(',')[7]))

acc9 = []
with open(path9, 'r') as f:
    for i in range(101):
        line = f.readline()
        if i>0:
            acc9.append(float(line.split(',')[7]))

plt.plot(x, acc1, 'o-', color='red', linewidth=1, label='Res-Unet++')
plt.plot(x, acc2, 'o-', color='cornflowerblue', linewidth=1, label='Unet')
plt.plot(x, acc3, 'o-', color='chartreuse', linewidth=1, label='Att-Unet')
plt.plot(x, acc4, 'o-', color='darkorange', linewidth=1, label='Unet++')
plt.plot(x, acc5, 'o-', color='saddlebrown', linewidth=1, label='R2-Unet')
plt.plot(x, acc6, '*-', color='purple', linewidth=1, label='RP-Unet')
plt.plot(x, acc7, 'o-', color='goldenrod', linewidth=1, label='R2Att-Unet')
plt.plot(x, acc8, 'o-', color='darkgrey', linewidth=1, label='YOLOv8n')
plt.plot(x, acc9, 'o-', color='aqua', linewidth=1, label='Mask-RCNN')

plt.xlabel('(a) Epoch', fontproperties = 'Times New Roman', size=18)
plt.ylabel('Accuracy', fontproperties = 'Times New Roman', size=18)
plt.legend(loc="best", frameon=False)  # 图例

plt.show()

