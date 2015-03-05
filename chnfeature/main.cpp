#include <iostream>
#include <fstream> 
#include <cmath>
#include <vector>
#include <typeinfo>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/opencv.hpp" 
#include <cv.h>
#include <cxcore.h> 
#include <cvaux.h>

#include "Pyramid.h"
#include "../misc/misc.hpp"

using namespace std;
using namespace cv;
void MultiImage_OneWin(const std::string& MultiShow_WinName, const vector<Mat>& SrcImg_V, CvSize SubPlot, CvSize ImgMax_Size)
{


	//************* Usage *************//
	//vector<Mat> imgs(4);
	//imgs[0] = imread("F:\\SA2014.jpg");
	//imgs[1] = imread("F:\\SA2014.jpg");
	//imgs[2] = imread("F:\\SA2014.jpg");
	//imgs[3] = imread("F:\\SA2014.jpg");
	//imshowMany("T", imgs, cvSize(2, 2), cvSize(400, 280));

	//Window's image
	Mat Disp_Img;
	//Width of source image
	CvSize Img_OrigSize = cvSize(SrcImg_V[0].cols, SrcImg_V[0].rows);
	//******************** Set the width for displayed image ********************//
	//Width vs height ratio of source image
	float WH_Ratio_Orig = Img_OrigSize.width/(float)Img_OrigSize.height;
	CvSize ImgDisp_Size = cvSize(100, 100);
	if(Img_OrigSize.width > ImgMax_Size.width)
		ImgDisp_Size = cvSize(ImgMax_Size.width, (int)(ImgMax_Size.width/WH_Ratio_Orig));
	else if(Img_OrigSize.height > ImgMax_Size.height)
		ImgDisp_Size = cvSize((int)(ImgMax_Size.height*WH_Ratio_Orig), ImgMax_Size.height);
	else
		ImgDisp_Size = cvSize(Img_OrigSize.width, Img_OrigSize.height);
	//******************** Check Image numbers with Subplot layout ********************//
	int Img_Num = (int)SrcImg_V.size();
	if(Img_Num > SubPlot.width * SubPlot.height)
	{
		cout<<"Your SubPlot Setting is too small !"<<endl;
		exit(0);
	}
	//******************** Blank setting ********************//
	CvSize DispBlank_Edge = cvSize(80, 60);
	CvSize DispBlank_Gap  = cvSize(15, 15);
	//******************** Size for Window ********************//
	Disp_Img.create(Size(ImgDisp_Size.width*SubPlot.width + DispBlank_Edge.width + (SubPlot.width - 1)*DispBlank_Gap.width, 
		ImgDisp_Size.height*SubPlot.height + DispBlank_Edge.height + (SubPlot.height - 1)*DispBlank_Gap.height), CV_8UC1);
	Disp_Img.setTo(0);//Background
	//Left top position for each image
	int EdgeBlank_X = (Disp_Img.cols - (ImgDisp_Size.width*SubPlot.width + (SubPlot.width - 1)*DispBlank_Gap.width))/2;
	int EdgeBlank_Y = (Disp_Img.rows - (ImgDisp_Size.height*SubPlot.height + (SubPlot.height - 1)*DispBlank_Gap.height))/2;
	CvPoint LT_BasePos = cvPoint(EdgeBlank_X, EdgeBlank_Y);
	CvPoint LT_Pos = LT_BasePos;

	//Display all images
	for (int i=0; i < Img_Num; i++)
	{
		//Obtain the left top position
		if ((i%SubPlot.width == 0) && (LT_Pos.x != LT_BasePos.x))
		{
			LT_Pos.x = LT_BasePos.x;
			LT_Pos.y += (DispBlank_Gap.height + ImgDisp_Size.height);
		}
		//Writting each to Window's Image
		Mat imgROI = Disp_Img(Rect(LT_Pos.x, LT_Pos.y, ImgDisp_Size.width, ImgDisp_Size.height));
		resize(SrcImg_V[i], imgROI, Size(ImgDisp_Size.width, ImgDisp_Size.height));

		LT_Pos.x += (DispBlank_Gap.width + ImgDisp_Size.width);
	}

	//Get the screen size of computer
	//int Scree_W = 1600;//GetSystemMetrics(SM_CXSCREEN);
	//int Scree_H = 900;//GetSystemMetrics(SM_CYSCREEN);

	cvNamedWindow(MultiShow_WinName.c_str(), CV_WINDOW_AUTOSIZE);
	//cvMoveWindow(MultiShow_WinName.c_str(),(Scree_W - Disp_Img.cols)/2 ,(Scree_H - Disp_Img.rows)/2);//Centralize the window
    IplImage ss(Disp_Img);
	cvShowImage(MultiShow_WinName.c_str(), &ss);
	cvWaitKey(0);
	cvDestroyWindow(MultiShow_WinName.c_str());
}

int main( int argc, char** argv)
{

    cv::TickMeter tk;
    for ( int c=0; c<5;c++ ) cout<<"a",cout<<"b";
	const int k=256; float R[k], G[k], B[k];
	if(size_t(R)&15)
		cout<<"bingo"<<endl;
	cout<<(size_t(R))<<endl;
	cout<<static_cast<const void*>(R)<<endl;



    Mat input_image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    //Mat input_image = imread("/media/yuanyang/disk1/git/adaboost/build/chnfeature/crop_000007.png", CV_LOAD_IMAGE_GRAYSCALE);

    imshow("input",input_image);
    cout<<"image size : "<<input_image.size()<<endl;
    feature_Pyramids ff1;
    vector<Mat> f_chns;
    tk.start();
    Mat fhog_feature;
    ff1.fhog( input_image, fhog_feature, f_chns, 0);
    tk.stop();
    cout<<"time consuming "<<tk.getTimeMilli()<<endl;
    cout<<"feature has "<<f_chns.size()<<" channels "<<endl;
    cout<<"size of the feature is "<<fhog_feature.size()<<endl;
    Mat draw;
    vector<Mat> zero_pi_channel( f_chns.begin() + 18, f_chns.begin()+27);
    cout<<"contrast unsensitive channels "<<zero_pi_channel.size()<<endl;
    ff1.visualizeHog( zero_pi_channel, draw);
    imshow("show", draw);
    waitKey(0);
    
    //vector<vector<Mat> > feature;
    //vector<double> scales;
    //vector<double> scalesw;
    //vector<double> scalesh;

    //vector<Mat> features;
    //ff1.computeChannels( input_image, features);

   // Mat mag,ori;
   // Mat gghist;
   // ff1.convt_2_luv( input_image, L, U, V);
   // ff1.convTri( L, smooth_input, 1, 3);
   // ff1.computeGradMag( L,U,V, mag, ori, false);

    //tk.start();
    //for(int c=0;c<10;c++)
    //    ff1.chnsPyramid( input_image, fealsl, scales, scalesw, scalesh);
    //tk.stop();
    //cout<<"opencv time -> "<<tk.getTimeMilli()/10<<endl;

    //saveMatToFile("opencvL.data", fealsl[0][0]);
    //saveMatToFile("opencvU.data", fealsl[0][1]);
    //saveMatToFile("opencvV.data", fealsl[0][2]);

    //tk.reset();tk.start();
    //for(int c=0;c<20;c++)
    //    ff1.chnsPyramid_sse( input_image, fealsl, scales, scalesw, scalesh);
    //tk.stop();
    //cout<<"sse time -> "<<tk.getTimeMilli()/20<<endl;

    //saveMatToFile("matlabL.data", fealsl[0][0]);
    //saveMatToFile("matlabU.data", fealsl[0][1]);
    //saveMatToFile("matlabV.data", fealsl[0][2]);

//    vector<Mat> features;
//    ff1.computeChannels( input_image, features);
//    saveMatToFile("p2.data", U);
//    saveMatToFile("p3.data", V);
      //  saveMatToFile("p1.data", fealsl[0][3]);
      //  saveMatToFile("p2.data", fealsl[0][4]);
      //  saveMatToFile("p3.data", fealsl[3][3]);
      //  saveMatToFile("p4.data", fealsl[3][4]);
//    saveMatToFile("p5.data", features[7]);
//    saveMatToFile("p6.data", features[8]);
//    saveMatToFile("p7.data", features[9]);
//	cout<<"number of scale is "<<feature.size()<<endl;
//	saveMatToFile("p3.data", feature[11][3]);
//	saveMatToFile("p4.data", feature[11][4]);
//	saveMatToFile("p5.data", feature[11][5]);
//	saveMatToFile("p6.data", feature[11][6]);
//	saveMatToFile("p7.data", feature[11][7]);
//	saveMatToFile("p8.data", feature[11][8]);
//	saveMatToFile("p9.data", feature[11][9]);


    //TickMeter tk;
    //Mat b_i, a_i;
    //tk.start();
    //resize( input_image, b_i, Size(), 0.25, 0.25, INTER_LINEAR);
    //tk.stop();
    //cout<<"time b is "<<tk.getTimeSec()<<endl;

    //tk.reset();tk.start();
    //resize( input_image, a_i, Size(), 0.25, 0.25, INTER_AREA);
    //tk.stop();
    //cout<<"time a is "<<tk.getTimeSec()<<endl;
    
    
    //for ( int c=0; c<feature.size(); c++) {
    //    cout<<"\n\n feature "<<c<<" is \n\n"<<feature[c]<<endl;
    //}

    //Mat test_img = imread("test.png");
    //int smooth = 5;

    //feature_Pyramids m_feature_gen;
	//
    //double *kern=new double[2*smooth+1];
	//for (int c=0;c<=smooth;c++)
	//{
	//	kern[c]=(double)((c+1)/((smooth+1.0)*(smooth+1.0)));
	//	kern[2*smooth-c]=kern[c];
	//}
	//Mat Km=Mat(1,(2*smooth+1),CV_64FC1,kern); 
    //Mat dst;
    //
    //vector< vector<Mat> > approPyramid;
    //vector<double> appro_scales;

    //m_feature_gen.chnsPyramid( test_img, approPyramid, appro_scales);
    //cout<<"chn 3 is "<<approPyramid[0][3].size()<<endl;
    //cout<<approPyramid[0][3]<<endl;
    //
    //m_feature_gen.convTri( approPyramid[0][3], dst, Km);
    //cout<<"dst size "<<dst.size()<<endl;
    //cout<<dst;

    //Mat norm_mag = approPyramid[0][3]/(dst+0.01);
    //cout<<"\n\nnorm_mag size "<<norm_mag.size()<<endl;
    //cout<<norm_mag;
	//
    //delete [] kern;
    return 0;
}
