#include <vector>
#include <string>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "scanner.h"


using namespace std;
using namespace cv;

int main( int argc, char** argv)
{
    scanner fhog_sc;
    fhog_sc.loadModel( string(argv[1]));
    /* test on one image */
    Mat img = imread( argv[2]);
    TickMeter tk;
    vector<Rect> results;
    vector<double> confs;
    tk.start();
    fhog_sc.detectMultiScale( img, results, confs, Size(40,40), Size(200,200), 1.2, 1);
    tk.stop();
    cout<<"detect time is "<<tk.getTimeMilli()<<endl;
    cout<<"size of results is "<<results.size()<<endl;
    for( int c=0;c<results.size();c++)
    {
        if( confs[c] < 1)
            continue;
        rectangle(  img, results[c], Scalar(255,0,0), 1);
        cout<<results[c]<<endl;
        cout<<"conf is "<<confs[c]<<endl;
    }
    imshow("result", img );
    waitKey(0);
    return 0;
}
