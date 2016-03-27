#face detect and localization
My demo for face detection and facial point localization. I reimplement the algorithm in [DLIB](http://dlib.net/), it now ***depends*** on ***OpenCV***, Boost(training phase only) only, you may find it useful in some case. 

you can choose [FHOG](http://120.52.73.79/arxiv.org/pdf/1502.00046.pdf) or OpenCV's haar detector. FHOG has a better ROC curve but takes a lot of time though... facial point localization takes ***4-5 ms*** on my computer, so it's pretty fast..

##demo
This is a demo image for the algorithm( 68 facial points, trained on Helen dataset), the localization part is fast and accurate enough in most situations ....
![sample](http://7xsc78.com1.z0.glb.clouddn.com/fhog_detect_sample.png)

##how to use
checkout and compile( CMake is the easist way, since it depends only on OpenCV in test time, you can easily set it up on Windows too)

### 1. FHOG + facial point localization
 see shape_predictor/test_main.cpp for example, download these two models: [fhog detector](http://pan.baidu.com/s/1qXTYQqo) and [facial_point_for_fhog](http://pan.baidu.com/s/1ge7W8y3), put them in the right folder, you are good to go.
 
### 2. Haar + facial point localization
see shape_predictor/test_haar_main.cpp, download [haar_detector](http://pan.baidu.com/s/1pLKNGfh) and [facial_point_for_haar](http://pan.baidu.com/s/1eSFay7C)

*note these 2 facial point models are not the same!*

##and ...
This also contains opencv implementation for *channels feature* and *soft cascade decision tree*, we use it for pedestrian detection, it works well! but I forget where I put the trained model ... 