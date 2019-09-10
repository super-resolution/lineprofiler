.. line_profiler documentation master file, created by
   sphinx-quickstart on Fri Jul 12 10:37:30 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to line profiler's documentation!
=========================================

Line profiler's purpose is to evaluate datasets in a biological/biophysical context. The software recognizes line shaped 
structures in image data and computes their mean position and orientation with sub-pixel accuracy. For each line a mean 
intensity profile is calculated. Utilising the whole image the software prevents biases, caused by preselecting suitable data
subsets. 

Currently line profiler support's two modes:

.. figure:: fig/Microtub.jpg
   
    **Microtuboli:** In Microtuboli mode profiles are by default fitted with the intensity projection of a cylinder, applying the theoretical values for inner, outer radius and resolution.

.. figure:: fig/SNC.jpg
   
    **SNC:** In SNC mode the profiles are collected in the first color channel while using the second channel for fitting. This mode allows the usage of the parameters upper and lower limit. Profiles featuring two maximas with a distance, exceeding the restrictions are excluded. As well as profiles without two maximas.



Line profiler's workflow contains multiple processing steps. The input image is convolved with a gaussian blur, 
compensating noise and intensity fluctuations. The skeletonize algorithm (1) reduces all expanded shapes to lines with 1 pixel width.
The function :ref:`compute_line_orientation<Utility package>` rearanges all pixels unequal zero, to continuous lines. The pixel 
coordinates of each line are fitted with a c-spline. The local derivative of the c-spline gives the direction 
for placing the line profiles. The averaged profiles for each line and the whole image are evaluated, by fitting the 
functions defined in :ref:`fitter<fitter>`.

.. image:: fig/FlowChartMicrotuboli.jpg
   :width: 98%



.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   controllers.utility
   controllers.fitter


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
