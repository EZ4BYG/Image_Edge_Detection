所有程序使用python3.8编写（python版本最好3.6以上），并需额外安装如下3个第三方库：
1. numpy —— 安装命令：pip install numpy
2. scipy —— 安装命令：pip install scipy
3. opencv —— 安装命令：pip install opencv-python

两种运行方式：
1. （推荐）使用PyCharm软件作为python程序的IDE，程序直接在PyCharm中运行即可；
2. 在程序所在路径下打开cmd，用命令运行，例如：python Canny.py

注意：
1. 测试图像与程序放在同一路径下时可直接运行程序，否则注意修改程序中图像的绝对路径；
2. 程序运行完毕会保存结果到当前路径下，若不想保存可注释掉所有程序中cv2.imwrite()语句；
3. Canny.py会调用Sobel.py中的函数，故这两个程序要放在同一路径下；
4. Canny.py程序有一些超参数可人为调整，包括：卷积核大小、高/低阈值。
