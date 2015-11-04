#include <iostream>
#include <stdlib.h>
#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "shape_predictor.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv)
{
	/* Load face detector*/
    CascadeClassifier face_detector;
	if(!face_detector.load("frontalface.xml"))
	{
		cout<<"Can not load model file "<<endl;
		return -2;
	}
	cout<<"Loading face detector done "<<endl;


	/* Load shape predictor */
	shape_predictor sp;
	if(!sp.load_model("haar_shape_model.xml"))
	{
		cout<<"Can not load shape predictor"<<endl;
		return 2;
	}

	/* Test */
	Mat input_img = imread("003764_29.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	//cv::resize( input_img, input_img, Size(0,0), 2, 2);
   
    vector<Rect> faces;
    vector<double> confs;
	face_detector.detectMultiScale(input_img, faces, 1.1, 3, 0, Size(60,60));
 
	vector<shape_type> shapes;
	for ( unsigned long i=0;i<faces.size(); i++)
	{
        cv::TickMeter tk;
        tk.start();
		shape_type shape = sp(input_img, faces[i]);
        tk.stop();
        cout<<"time "<<tk.getTimeMilli()<<endl;
		shapes.push_back( shape );

		Mat rotate_face;
		shape_predictor::align_face(shape, input_img, 128, rotate_face);
		imshow("rotate", rotate_face);
		waitKey(0);
	}
	
	Mat draw = shape_predictor::draw_shape(input_img, faces, shapes);
	imshow("show", draw);
	waitKey(0);
	return 0;
}
