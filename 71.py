import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.sans-serif"]="SimHei"

img = Image.open("lena.tiff")
img_r,img_g,img_b = img.split()
plt.figure(figsize=(10,10))
plt.suptitle("图像基本操作",fontsize=20,color="blue")

plt.subplot(221)
plt.axis("off")
imgr = img_r.resize((50,50))
plt.imshow(imgr,cmap="gray")
plt.title("R-缩放",fontsize=14)

plt.subplot(222)
imgg = img_g.transpose(Image.FLIP_LEFT_RIGHT)
imggg = imgg.transpose(Image.ROTATE_270)
plt.imshow(imggg,cmap="gray")
plt.title("G-镜像+旋转",fontsize=14)

plt.subplot(223)
plt.axis("off")
imgb = img_b.crop((0,0,300,300))
plt.imshow(imgb,cmap="gray")
plt.title("B-裁剪",fontsize=14)

img_rgb=Image.merge("RGB",[img_r,img_g,img_b])
plt.subplot(224)
plt.axis("off")
plt.imshow(img_rgb)
plt.title("RGB",fontsize=14)
img_rgb.save("text.png")

plt.show()