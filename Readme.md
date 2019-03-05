# Line Profiler

To evaluate our results we constructed an automated image processing software. The program is called Line Profiler and is available as executable on:
Zenodo… 
The source code can be found here:
https://github.com/super-resolution/line_profiler. 
In a first step Line Profiler uses several image processing algorithms to select potential regions of interest. The SNC Helix structure is spotted by detecting closed shapes in the image. This is achieved by blurring the image with a nxn Gaussian Kernel, to account for potential holes in the line. A threshold algorithm (1) converts the grayscale to a binary image. A floodfill algorithm leaves only the areas embedded in closed shapes. The maximum value of a distance Transformation (2) reveals the candidate points where the helix structure is in plain. The direction of the applied line profile is determined by the gradient direction arctan(SobelX, SobelY) of the edge point (3) closest to the preselected candidate point.

##Features:
1. Find appropriate line profiles <br />
![alt text](https://github.com/super-resolution/line_profiler/blob/master/images/MIP.png)
2. Get a histogram of the SNC distance <br />
![alt text](https://github.com/super-resolution/line_profiler/blob/master/images/Histogram.png)
3. Mean of all line profiles <br />
![alt text](https://github.com/super-resolution/line_profiler/blob/master/images/profiles.png)

## References:
(1) Nobuyuki Otsu: A threshold selection method from grey level histograms. In: IEEE Transactions on Systems, Man, and Cybernetics. New York, 9.1979, S. 62–66. ISSN 1083-4419
(2) Felzenszwalb, Pedro F. and Huttenlocher, Daniel P. Distance Transforms of Sampled Functions, TR2004-1963, TR2004-1963 (2004)
(3) JOHN CANNY,A Computational Approach to Edge Detection, Editor(s): Martin A. Fischler, Oscar Firschein, Readings in Computer Vision, Morgan Kaufmann, 1987, Pages 184-203, ISBN 9780080515816, https://doi.org/10.1016/B978-0-08-051581-6.50024-6.

