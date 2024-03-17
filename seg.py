import cv2

# 读取原始图像
img = cv2.imread(r'E:\github\pytorch-nested-unet-master\pytorch-nested-unet-master\DMI_OD_UPPER_REFLECTED_IR_19756451.JPG')

# 创建掩膜
mask = cv2.imread(r'E:\github\pytorch-nested-unet-master\pytorch-nested-unet-master\DMI_OD_UPPER_REFLECTED_IR_19756451.png', cv2.IMREAD_GRAYSCALE)

# 使用掩膜抠图
result = cv2.bitwise_and(img, img, mask=mask)

# 显示结果
cv2.imshow('Result', result)
cv2.imwrite('1.png', result)
cv2.waitKey(0)
cv2.destroyAllWindows()