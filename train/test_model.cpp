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
	/*  test the svd function */
	Mat g = (Mat_<double>(3,3)<<0.011343736558495, 
								0.083819505802211,
								0.011343736558495,
								0.083819505802211,
								0.619347030557177,
								0.083819505802211,
								0.011343736558495,
								0.083819505802211,
								0.011343736558495);
	Mat w,u,vt;
	SVD::compute(g, w, u , vt);
	cout<<"g's type is "<<g.type()<<endl;
	cout<<"w's type is "<<w.type()<<endl;
	cout<<"u's type is "<<u.type()<<endl;
	cout<<"v's type is "<<vt.type()<<endl;

	cout<<"w is \n"<<w<<endl;
	cout<<"u is \n"<<u<<endl;
	cout<<"vt is \n"<<vt<<endl;

	cout<<" g is \n"<<g<<endl;
	cout<<"get g back \n"<<u.col(0)*w.col(0).row(0)*vt.row(0)<<endl;

	double min_w, max_w;
	cv::Point min_p,max_p;
	minMaxLoc( w, &min_w, &max_w, &min_p, &max_p );
	cout<<"min_w is "<<min_w<<", max_w is "<<max_w<<" min_p is "<<min_p<<", max_p is "<<max_p<<endl;
	

    string model_path = argv[1];
    string test_img_folder = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/FDDB/test/imgs/";
    string test_img_gt = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/FDDB/test/gts/";
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

    /*  use check_detector to evaluate the performance */
    double _hit = 0;
    double _FPPI =0;
    detect_check<scanner> dc;
    dc.set_path( test_img_folder, test_img_gt, "", true);
    dc.set_parameter( Size(40,40), Size(300,300), 1.2, 1, 0);

    //vector<double> hits;
    //vector<double> fppis;
    //dc.generate_roc( fhog_sc, fppis, hits);
    dc.test_detector( fhog_sc, _hit, _FPPI);
    dc.get_stat_on_missed();
    cout<<"Results : \nHit : "<<_hit<<endl<<"FPPI : "<<_FPPI<<endl;

    return 0;
}
