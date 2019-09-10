# Line Profiler
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2643214.svg)](https://doi.org/10.5281/zenodo.2643214) <br />
Line profiler’s purpose is to evaluate datasets in a biological/biophysical context. The software recognizes line shaped structures in image data and computes their mean position and orientation with sub-pixel accuracy. For each line a mean intensity profile is calculated. Utilising the whole image the software prevents biases, caused by preselecting suitable data subsets. <br />
A detailed explanation and documentation can be found ![here](https://line-profiler.readthedocs.io/en/latest/).
##Installation
An executable can be found on zenodo.
## Features:
1. Find appropriate line profiles <br />
![alt text](https://github.com/super-resolution/line_profiler/blob/master/images/MIP.png)
2. Get a histogram of the SNC distance <br />
![alt text](https://github.com/super-resolution/line_profiler/blob/master/images/Histogram.png)
3. Mean of all line profiles <br />
![alt text](https://github.com/super-resolution/line_profiler/blob/master/images/profiles.png)

## References:
(1) Nobuyuki Otsu: A threshold selection method from grey level histograms. In: IEEE Transactions on Systems, Man, and Cybernetics. New York, 9.1979, S. 62–66. ISSN 1083-4419 <br />
(2) Felzenszwalb, Pedro F. and Huttenlocher, Daniel P. Distance Transforms of Sampled Functions, TR2004-1963, TR2004-1963 (2004) <br />
(3) JOHN CANNY,A Computational Approach to Edge Detection, Editor(s): Martin A. Fischler, Oscar Firschein, Readings in Computer Vision, Morgan Kaufmann, 1987, Pages 184-203, ISBN 9780080515816, https://doi.org/10.1016/B978-0-08-051581-6.50024-6. <br />

