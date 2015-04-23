#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <ctime>


#include "boost/filesystem.hpp"
#include "boost/lambda/bind.hpp"

#include "../misc/misc.hpp"
#include "../chnfeature/Pyramid.h"
#include "../scanner/scanner.h"

#include <omp.h> 

#include "detect_check.h"

#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

namespace bf = boost::filesystem;
namespace bl = boost::lambda;

int main( int argc , char ** argv)
{
    string model_path(argv[1]);
    //string model_path2 = argv[2];
    //string model_path3 = argv[3];
    //string model_path4 = argv[4];
    //string model_path5 = argv[5];
    vector<string> m_paths;
    m_paths.push_back( model_path);
    //m_paths.push_back( model_path2);
    //m_paths.push_back( model_path3);
    //m_paths.push_back( model_path4);
    //m_paths.push_back( model_path5);

    //string test_img_folder = "/media/yuanyang/disk1/data/face_detection_database/PCI_test/imgs/";
    string test_img_folder = "/media/yuanyang/disk1/data/goutiantian_noise/pos/";
    //string test_img_gt = "/media/yuanyang/disk1/data/face_detection_database/PCI_test/gts/";
    string test_img_gt = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/FDDB/test/gts/";

    TickMeter tk;

    scanner fhog_sc;
    if( !fhog_sc.loadModels( m_paths ) )
    {
        cout<<"Can not load the model file "<<endl;
        return -1;
    }

    /*  show the detector */
    fhog_sc.visualizeDetector();


    /*  test on a giving image */
    //Mat frame_image = imread( "/media/yuanyang/disk1/git/fhog_pyramid_detect/build/train/image_0044.png" );
    //vector<Rect> results;
    //vector<double> confs;
    //tk.reset();tk.start();
    //fhog_sc.detectMultiScale( frame_image, results, confs, Size(80,80), Size(1000,1000),1.2, 1, 0);
    //tk.stop();
    //cout<<"processing time is "<<tk.getTimeMilli()<<endl;

    //for(unsigned int c=0;c<results.size();c++)
    //    rectangle( frame_image, results[c], Scalar(255,0,255), 2);
    //imshow("result",frame_image);
    //waitKey(0);

    /* test on a folder */
    //detect_check<scanner> dc;
    //dc.set_path( test_img_folder, test_img_gt, "", true);
    //dc.set_parameter( Size(80,15), Size(320,60), 1.2, 1, 0);
    //dc.show_results( fhog_sc );


    /*  1 test on a giving video */
    string video_name = string(argv[2]);
    cv::VideoCapture vc( video_name);
    if(!vc.isOpened())
    {
        cout<<"Can not open the video file "<<string(argv[2])<<endl;
    }

    bool do_detection = false;
    while(true)
    {
        Mat frame_image;
        vc>>frame_image;
        if(frame_image.empty())
        {
            cout<<"video ends "<<endl;
            break;
        }
        resize( frame_image, frame_image, Size(0,0), 0.8, 0.8);

        /*  do detection */
        if(do_detection)
        {
            vector<Rect> results;
            vector<double> confs;
            tk.reset(); tk.start();
            fhog_sc.detectMultiScale( frame_image, results, confs, Size(80,80), Size(600,600),1.2, 1, -0.08);
            tk.stop();
            cout<<"processing time is "<<tk.getTimeMilli()<<endl;
            for(unsigned int c=0;c<results.size();c++)
            {
                cout<<"conf is "<<confs[c]<<endl;
                rectangle( frame_image, results[c], Scalar(255,0,255), 2);
            }
        }

        imshow("frame", frame_image);
        char c = waitKey(30);
        if(c=='t')
            do_detection = true;
        else if(c=='s')
            do_detection = false;

    }



    /* 2 test on FDDB face detection database
     * ------------------------------------------------------------------------------------
     */
    /*  use check_detector to evaluate the performance */
    //double _hit = 0;
    //double _FPPI =0;
    //detect_check<scanner> dc;
    //dc.set_path( test_img_folder, test_img_gt, "", true);
    //dc.set_parameter( Size(80,80), Size(400,400), 1.2, 1, 0);

    //vector<double> hits;
    //vector<double> fppis;
    ////dc.generate_roc( fhog_sc, fppis, hits, 1, -1.5);
    //dc.test_detector( fhog_sc, _hit, _FPPI);
    //dc.get_stat_on_missed();
    //cout<<"Results : \nHit : "<<_hit<<endl<<"FPPI : "<<_FPPI<<endl;
    //fhog_sc.saveModel( "f_rl_rr_face.xml","pin le lao ming");
    /*
     * ------------------------------------------------------------------------------------
     */

    return 0;
}
