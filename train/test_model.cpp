#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <ctime>

#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "boost/filesystem.hpp"
#include "boost/lambda/bind.hpp"

#include "../misc/misc.hpp"
#include "../chnfeature/Pyramid.h"
#include "../svm/opencv_warpper_libsvm.h"
#include "../scanner/scanner.h"

#include <omp.h> 

#include "detect_check.h"

#define SAVE_IMAGE

using namespace std;
using namespace cv;

namespace bf = boost::filesystem;
namespace bl = boost::lambda;

int main( int argc , char ** argv)
{
    string model_path = argv[1];
    string test_img_folder = "/media/yuanyang/disk1/data/face_detection_database/PCI_test/imgs/";
    string test_img_gt = "/media/yuanyang/disk1/data/face_detection_database/PCI_test/gts/";
    double detect_threshold = -0.8;

    TickMeter tk;

    scanner fhog_sc;
    if( !fhog_sc.loadModel( model_path ) )
    {
        cout<<"Can not load the model file "<<endl;
        return -1;
    }

    /*  show the detector */
    fhog_sc.visualizeDetector();


    /* test once */
    Mat input_image = imread( argv[2]);
    vector<Rect> dets;
    vector<double> confs;
    fhog_sc.detectMultiScale( input_image, dets, confs, Size(30,30),Size(500,500), 1.2,1, -0.2 );
    cout<<"det "<<dets.size()<<endl;
    
    for ( unsigned int c=0;c<dets.size() ;c++ ) {
        rectangle( input_image, dets[c], Scalar(255,0,128), 2);
    }
    imshow("show", input_image);
    waitKey(0);

    /*  use check_detector to evaluate the performance */
    double _hit = 0;
    double _FPPI =0;
    detect_check<scanner> dc;
    dc.set_path( test_img_folder, test_img_gt, "", true);
    dc.set_parameter( Size(30,30), Size(400,400), 1.2, 1, -1);

    vector<double> hits;
    vector<double> fppis;
    dc.generate_roc( fhog_sc, fppis, hits, 1, -1.5);
    dc.test_detector( fhog_sc, _hit, _FPPI);
    dc.get_stat_on_missed();
    cout<<"Results : \nHit : "<<_hit<<endl<<"FPPI : "<<_FPPI<<endl;

    return 0;
}
