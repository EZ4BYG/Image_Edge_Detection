# Edge_Detection
Classical edge-detection algorithms using python

## 1. Edge-detection operators based on first derivative
- *Roberts.py*
- *Prewitt.py* and *Scharr.py*
- *Kirsch.py* and *Robinson.py* —— Update: Robinson operator included in Kirsch.py
- *Sobel.py* and *Sobel_Normalize.py* —— good algorithm
- *Canny.py* —— best algorithm. We can assume that the problem of edge-detection has been solved by this algorithm.

Tip1: Algorithms using first-derivative information are to detect the positions of gray-value change in the image.

## 2. Edge-detection operators based on second derivative
- *Laplacian.py*
- *LoG.py*
 
Tip2：Various algorithms (just list 2 algorithms here) that use the second-derivative information are not designed to exceed the above algorithms (first-derivative), but rather they can detect some special boundary —— The positions of the pixel where the gray-value changes the fastest!

## Necessary python packages
- *Numpy* —— pip install numpy
- *Scipy* —— pip install scipy 
- *opencv* —— pip install opencv-python

Tip3: python > 3.6. Codes are best opened and run with PyCharm IDE. 
