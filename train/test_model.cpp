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
    string model_path2 = argv[2];
    string model_path3 = argv[3];
    string model_path4 = argv[4];
    string model_path5 = argv[5];
    vector<string> m_paths;
    m_paths.push_back( model_path);
    m_paths.push_back( model_path2);
    m_paths.push_back( model_path3);
    m_paths.push_back( model_path4);
    m_paths.push_back( model_path5);

    string test_img_folder = "/media/yuanyang/disk1/data/face_detection_database/PCI_test/imgs/";
    //string test_img_folder = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/FDDB/test/imgs/";
    string test_img_gt = "/media/yuanyang/disk1/data/face_detection_database/PCI_test/gts/";
    //string test_img_gt = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/FDDB/test/gts/";

    TickMeter tk;

    scanner fhog_sc;
    if( !fhog_sc.loadModels( m_paths ) )
    {
        cout<<"Can not load the model file "<<endl;
        return -1;
    }

    /*  show the detector */
    fhog_sc.visualizeDetector();


    /*  use check_detector to evaluate the performance */
    double _hit = 0;
    double _FPPI =0;
    detect_check<scanner> dc;
    dc.set_path( test_img_folder, test_img_gt, "", true);
    dc.set_parameter( Size(80,80), Size(400,400), 1.2, 1, 0);

    vector<double> hits;
    vector<double> fppis;
    //dc.generate_roc( fhog_sc, fppis, hits, 1, -1.5);
    dc.test_detector( fhog_sc, _hit, _FPPI);
    dc.get_stat_on_missed();
    cout<<"Results : \nHit : "<<_hit<<endl<<"FPPI : "<<_FPPI<<endl;

    return 0;
}
