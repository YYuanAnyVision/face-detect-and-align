#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "boost/filesystem.hpp"

#include "misc.hpp"
#include "jitterImage.h"

using namespace std;
using namespace cv;



namespace bf = boost::filesystem;

int main( int argc, char** argv)
{

    FileStorage fs("testMat.xml", FileStorage::WRITE);
    vector<Mat> mats;
    Mat a = Mat::zeros(4,5,CV_32F);
    Mat b = Mat::zeros(9,1, CV_64F);
    mats.push_back(a);
    mats.push_back(b);
    fs<<"mats"<<mats;
    fs.release();

	//Mat input_image  = imread("../../data/test.png");

	//cout<<"input image, cols "<<input_image.cols<<" rows "<<input_image.rows<<endl;
	//imshow("original", input_image);
	
	// 1 resize
	//cout<<"test resizeBbox "<<endl;
	//Rect o1(100, 100, 150, 150);
	//Rect o1_resize = resizeBbox( o1, 1, 0.5);
	//Rect o2_resize = resizeBbox( o1, 0.5, 1);
	//imshow( "before resize", input_image(o1));
	//imshow( "resize o1", input_image(o1_resize) );
	//imshow( "resize o2", input_image(o2_resize) );

	//cout<<"test using cropSize "<<endl;
	//Rect cropEnlarge( 400, 200, 200, 250);
	//imshow("cropSizeimage", cropImage(input_image, cropEnlarge));
	
	// 2 sample box 
	//vector<Rect> pos;
	//sampleRects( 30, input_image.size(), cv::Size(150,80), pos);
	//cout<<"size of pos is "<<pos.size()<<endl;
	//for( int c=0;c<pos.size(); c++)
	//{
	//	cout<<"c is "<<c<<" rect is "<<pos[c]<<endl;
	//	imshow( "samples", input_image( pos[c] ) );
	//	waitKey(30);
	//}
		
	// 3 jitter image
	
	//vector<Mat> outs;

	//jitterImage( input_image, outs, input_image.size(), 12, true, 4, 50, 4, 30);
	//cout<<"the size of outs is "<<outs.size()<<endl;
	//for( int c=0;c<outs.size();c++)
	//{
	//	imshow("samples", outs[c]);
	//	waitKey(0);
	//}
	
	string _image_path = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/pos/";
	string _gt_path    = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/posGt_opencv/";

	bf::path image_path( _image_path );

	if(!bf::exists( image_path))
	{
		cout<<"not a path "<<image_path<<endl;
	}

	bf::directory_iterator end_it;
	int counter = 0;
	for( bf::directory_iterator file_iter(image_path); file_iter != end_it; file_iter++ )
    {
		bf::path s = *(file_iter);
        string basename = bf::basename( s );
        string pathname = file_iter->path().string();
		string extname  = bf::extension( s );
		
		Mat im = imread( pathname );
		string gt_file_path = _gt_path + basename + ".txt";
		FileStorage fs( gt_file_path, FileStorage::READ | FileStorage::FORMAT_XML);
		vector<Rect> rrs;
		fs["boxes"]>> rrs;

			
		for ( int c=0; c<rrs.size();c++ ) 
		{
			rrs[c] = resizeToFixedRatio( rrs[c], 0.41, 1);
			cout<<"ratio is "<<rrs[c].width*1.0/rrs[c].height<<endl;
			rectangle( im, rrs[c], Scalar(255,0,0), 2);
		}

		imshow("im", im );
		waitKey(0);
	}


	waitKey(0);
	return 0;
}
