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

using namespace std;
using namespace cv;

namespace bf = boost::filesystem;
namespace bl = boost::lambda;

int main( int argc , char ** argv)
{
    string model_path = "../train/scanner.xml";
    string test_img_folder = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/FDDB/imgs/";
    TickMeter tk;

    scanner fhog_sc;
    if( !fhog_sc.loadModel( model_path ) )
    {
        cout<<"Can not load the model file "<<endl;
        return -1;
    }

    bf::directory_iterator test_path( test_img_folder);
    bf::directory_iterator end_it;
    for( bf::directory_iterator file_iter( test_path ); file_iter!=end_it; file_iter++)
	{
        string pathname = file_iter->path().string();
		string extname  = bf::extension( *file_iter);
		if( extname!=".jpg" && extname!=".bmp" && extname!=".png" &&
			extname!=".JPG" && extname!=".BMP" && extname!=".PNG")
			continue;
        Mat img = imread( pathname );
        vector<Rect> det_results;
        vector<double> det_confs;
        tk.reset();tk.start();
        fhog_sc.detectMultiScale( img, det_results, det_confs, Size(40,40),Size(400,400),1.1,1);
        tk.stop();cout<<"Detect time is "<<tk.getTimeMilli()<<" ms"<<endl;
        for( int c=0;c<det_results.size();c++)
        {
            if( det_confs[c] < 1 )
                continue;
            rectangle( img, det_results[c], Scalar(188, 45, 213), 2);
            cout<<"det_result "<<det_results[c]<<", conf is "<<det_confs[c]<<endl;
        }
        cout<<endl;
        imshow("show", img);
        waitKey(30);
	}

    return 0;
}
