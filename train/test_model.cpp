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



//#define SAVE_IMAGE

using namespace std;
using namespace cv;

namespace bf = boost::filesystem;
namespace bl = boost::lambda;

bool isSameTarget( Rect r1, Rect r2)
{
	Rect intersect = r1 & r2;
	if(intersect.width * intersect.height < 1)
		return false;
	
	double union_area = r1.width*r1.height + r2.width*r2.height - intersect.width*intersect.height;

	if( intersect.width*intersect.height/union_area < 0.5 )
		return false;

	return true;
}

int main( int argc , char ** argv)
{
    string model_path = argv[1];
    string test_img_folder = "/media/yuanyang/disk1/libs/dlib-18.10/examples/faces/";
    string test_img_gt = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/FDDB/convert_opencv/";

    TickMeter tk;

    scanner fhog_sc;
    if( !fhog_sc.loadModel( model_path ) )
    {
        cout<<"Can not load the model file "<<endl;
        return -1;
    }

    /*  show the detector */
    fhog_sc.visualizeDetector();
    
    /*  store the paths */
    bf::directory_iterator test_path( test_img_folder);
    bf::directory_iterator end_it;
    vector<string> image_path_vector;
    vector<string> gt_path_vector;
    for( bf::directory_iterator file_iter( test_path ); file_iter!=end_it; file_iter++)
	{
        string pathname = file_iter->path().string();
        string basename = bf::basename( *file_iter);
		string extname  = bf::extension( *file_iter);
		if( extname!=".jpg" && extname!=".bmp" && extname!=".png" &&
			extname!=".JPG" && extname!=".BMP" && extname!=".PNG")
			continue;
        bf::path tmp_gt_path( test_img_gt + basename + ".xml");
        if(!bf::exists(tmp_gt_path))
            continue;
        image_path_vector.push_back( pathname);
        gt_path_vector.push_back( test_img_gt + basename + ".xml");
	}
    
    int number_of_target = 0;
    int number_of_fn = 0;
    int number_of_wrong = 0;
    int Nthreads = omp_get_max_threads();
    //#pragma omp parallel for num_threads(Nthreads) reduction( +: number_of_fn) reduction( +: number_of_wrong ) reduction(+:number_of_target)
    for( int i=0;i<image_path_vector.size(); i++)
	{ 
		// reading groundtruth...
        cout<<"processing image "<<image_path_vector[i]<<endl;
		vector<Rect> target_rects;
        FileStorage fst( gt_path_vector[i], FileStorage::READ | FileStorage::FORMAT_XML);
        fst["boxes"]>>target_rects;
        fst.release();
        number_of_target += target_rects.size();

		// reading image
		Mat test_img = imread( image_path_vector[i]);
		vector<Rect> det_rects;
		vector<double> det_confs;
        fhog_sc.detectMultiScale( test_img, det_rects, det_confs, Size(40,40),Size(300,300),1.2,1,1);


        /* debug show */
        for ( int c=0;c<det_rects.size() ; c++) {
            rectangle( test_img, det_rects[c], Scalar(0,0,255), 2);
            cout<<"conf is "<<det_confs[c]<<endl;
        }
        cout<<endl;
        imshow("test", test_img);
        waitKey(0);

		int matched = 0;
		vector<bool> isMatched_r( target_rects.size(), false);
		vector<bool> isMatched_l( det_rects.size(), false);
		for( int c=0;c<det_rects.size();c++)
		{
			for( int k=0;k<target_rects.size();k++)	
			{
				if( isSameTarget( det_rects[c], target_rects[k]) && !isMatched_r[k] && !isMatched_l[c])
				{
					matched++;
					isMatched_r[k] = true;
					isMatched_l[c] = true;
					break;
				}
			}
		}

        bf::path t_path( image_path_vector[i]);
        string basename = bf::basename(t_path);
        //#pragma omp critical
        {
            for(int c=0;c<isMatched_r.size();c++)
            {
                if( !isMatched_r[c])
                {
                    number_of_fn++;
#ifdef SAVE_IMAGE
                    stringstream ss;ss<<c;string index_string;ss>>index_string;
                    string save_path = "pos_fn/"+basename+"_"+index_string+".jpg";
                    Mat save_img = cropImage( test_img, target_rects[c] );
                    imwrite( save_path, save_img);
#endif
                }
            }

            for(int c=0;c<isMatched_l.size();c++)
            {
                if( !isMatched_l[c])
                {
                    number_of_wrong++;
#ifdef SAVE_IMAGE
                    stringstream ss;ss<<c;string index_string;ss>>index_string;
                    string save_path = "pos_fp/"+basename+"_"+index_string+".jpg";
                    Mat save_img = cropImage( test_img, det_rects[c]);
                    imwrite( save_path, save_img);
#endif
                }
            }
        }
	}
    cout<<"number of targets is "<<number_of_target<<endl;
    cout<<"number of fn is "<<number_of_fn<<endl;
    cout<<"number of fp is "<<number_of_wrong<<endl;

    cout<<"hit --- > "<<1.0*(number_of_target - number_of_fn)/number_of_target<<endl;
    cout<<"FPPI --- > "<<1.0*(number_of_wrong)/image_path_vector.size()<<endl;
    return 0;
}
