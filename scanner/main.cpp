#include <vector>
#include <string>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "scanner.h"


using namespace std;
using namespace cv;

int main( int argc, char** argv)
{
    FileStorage fs("svm_weight.xml", FileStorage::READ);
    Mat weight_vector;
    fs["svm_weight"]>>weight_vector;
    
    scanner fhog_sc;
    fhog_sc.setParameters( 8, 9, Size(80,80), Size(96,96), weight_vector );

    /* test on one image */
    Mat img = imread( argv[1]);
    vector<Rect> results;
    vector<float> confs;
    fhog_sc.slide_image( img, results, confs, 1);
    cout<<"size of results is "<<results.size()<<endl;
    for( int c=0;c<results.size();c++)
    {
        rectangle(  img, results[c], Scalar(255,0,0), 1);
        imshow("result", img );
        cout<<"conf is "<<confs[c]<<endl;
        waitKey(0);
    }
    return 0;
}
