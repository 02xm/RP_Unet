import matplotlib.pyplot as plt
import numpy as np

# data-unet,unet++,att-unet,r2-unet,r2att-unet,resUnet++,yolov8,mask-rcnn,paper
time = [0.25,0.31,0.43,0.61,0.53,0.34,0.08,0.20,0.19]
model_size = [29.978,34.983,133.111,149.174,150.523,15.528,9.8,245,45.359]
color=['g', 'deepskyblue', 'c', 'k', 'm', 'y', 'cornflowerblue', 'b', 'r']
labels=['Unet', 'Unet++', 'Att-Unet', 'R2-Unet', 'R2Att-Unet', 'Res-Unet++', 'YOLOv8', 'Mask-RCNN', 'RP-Unet']
p = []
for i in range(9):
       p1=plt.scatter(time[i], model_size[i], s=12 ** 2, c=color[i])
       p.append(p1)
       plt.annotate(model_size[i], xy = (time[i], model_size[i]), xytext = (time[i]-0.02, model_size[i]+15))

# it depends on the data distribuion
plt.xlim(0, 0.6)
plt.xticks([0,0.2,0.40,0.60], size=10),
plt.ylim(0, 300)
plt.yticks(range(0,300,50), size=10)

plt.xlabel('Time(S)', fontproperties = 'Times New Roman', size=18)
plt.ylabel('Model Size(M)', fontproperties = 'Times New Roman', size=18)

legend_font = {"family" : "Times New Roman", "size": 15}
legend=plt.legend(p, labels, loc='upper left', frameon=False, prop=legend_font, labelspacing=0.25, handletextpad=0.1)
# set 100 of legend shape
for handle in legend.legend_handles:
        handle.set_sizes([100])

plt.show()
