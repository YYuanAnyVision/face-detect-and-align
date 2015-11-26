#include <iostream>
#include <stdlib.h>
#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "../scanner/scanner.h"
#include "shape_predictor.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv)
{
	/* Load face detector*/
	scanner fhog_sc;
	if(!fhog_sc.loadModel("super_pack_lfw.xml"))
	{
		cout<<"Can not load face detector .."<<endl;
		return 1;
	}

	/* Load shape predictor */
	shape_predictor sp;
	if(!sp.load_model("model.xml"))
	{
		cout<<"Can not load shape predictor"<<endl;
		return 2;
	}

	/* Test */
	Mat input_img = imread( argv[1] ,CV_LOAD_IMAGE_GRAYSCALE);
	//cv::resize( input_img, input_img, Size(0,0), 2, 2);
	vector<Rect> faces;
	vector<double> confs;
	fhog_sc.detectMultiScale( input_img, faces, confs, Size(80,80), Size(300,300), 1.2, 1, 0);
	
    cout<<"show ..."<<endl;
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

        /*  1 old align method */
		//shape_predictor::align_face(shape, input_img, 128, rotate_face);

        /*  2 new align method */
        shape_predictor::align_face_new(  shape, input_img, rotate_face, 256, 0.2);

		imshow("rotate", rotate_face);
		waitKey(0);
	}
	
	Mat draw = shape_predictor::draw_shape(input_img, faces, shapes);
	imshow("show", draw);
	waitKey(0);
	return 0;
}
