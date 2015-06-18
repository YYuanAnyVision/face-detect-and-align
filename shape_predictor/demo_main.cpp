#include <iostream>
#include <stdlib.h>
#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include "../scanner/scanner.h"
#include "shape_predictor.hpp"
#include "tinyxml.h"

using namespace std;
using namespace cv;

double isSameTarget( Rect r1, Rect r2)
{
	Rect intersect = r1 & r2;
	if(intersect.width * intersect.height < 1)
		return 0;

	double union_area = r1.width*r1.height + r2.width*r2.height - intersect.width*intersect.height;

	return intersect.width*intersect.height/union_area;
}


string yy_to_string( int x)
{
	stringstream ss;
	ss<<x;
	string str;
	ss>>str;
	return str;
}

void load_train_file( 
	scanner &face_det,				/* in : face detector */
	const string &train_file_path,	/* in : train file path*/
	const string &image_root,		/* in : root of the image folder*/
	int face_width,			        /* in : resize those whos width > image_max_width*/
	vector<Mat> &imgs,				/* out: imgs*/
	vector<vector<Rect> > &rects,	/* out: rects */
	vector<vector<shape_type> > &shapes)	/* out: shapes */
{
	double t_scale = 1.0;
	imgs.clear();
	rects.clear();
	shapes.clear();

	/* Load the training data form xml file*/
	TiXmlDocument inputDoc( train_file_path);
	if(!inputDoc.LoadFile(TIXML_ENCODING_UNKNOWN))
	{
		cout<<"Can not load xml file "<<train_file_path<<endl;
		return;
	}
	TiXmlElement *rootElement = inputDoc.RootElement();	// dataset section
	TiXmlElement *nameElement = rootElement->FirstChildElement(); // name section
	//cout<<nameElement->Value()<<":\n"<<nameElement->GetText()<<endl;
	TiXmlElement *commentElement = nameElement->NextSiblingElement(); // comment section
	//cout<<commentElement->Value()<<":\n"<<commentElement->GetText()<<endl;

	TiXmlElement *imagesElement = commentElement->NextSiblingElement();
	TiXmlElement *iterImageElement = imagesElement->FirstChildElement();

	long counter =0;
	for(; iterImageElement; iterImageElement = iterImageElement->NextSiblingElement()) //image section
	{
		cout<<"processing image "<<counter++<<endl;
		TiXmlAttribute *iterAtt = iterImageElement->FirstAttribute();
		// eg file = 2008.12.23.jpg
		//cout<<iterAtt->Name()<<" = "<<iterAtt->Value()<<endl;
		t_scale = 1.0;
		bool first_rect = true;
		bool matched = false;
		Mat input_img = imread( image_root + iterAtt->Value(), CV_LOAD_IMAGE_GRAYSCALE);
		
		assert( !input_img.empty());

		vector<Rect> rects_in_this_image;
		vector<shape_type> shapes_in_this_image;

		TiXmlElement *iterBox = iterImageElement->FirstChildElement();	// box section
		for (; iterBox; iterBox = iterBox->NextSiblingElement())
		{
			Rect temp_rect;
			TiXmlAttribute *iterRect = iterBox->FirstAttribute();
			temp_rect.y = atoi(iterRect->Value()); iterRect = iterRect->Next();
			temp_rect.x = atoi(iterRect->Value()); iterRect = iterRect->Next();
			temp_rect.width = atoi(iterRect->Value()); iterRect = iterRect->Next();
			temp_rect.height = atoi(iterRect->Value()); iterRect = iterRect->Next();
			if ( first_rect )
			{
				t_scale = 1.0*face_width/temp_rect.width;
				first_rect = false;

				/* add the image*/
				cv::resize( input_img, input_img, Size(0,0), t_scale, t_scale);
				
			}

			temp_rect.y *= t_scale;
			temp_rect.x *= t_scale;
			temp_rect.width *= t_scale;
			temp_rect.height *= t_scale;

			/*show */
			//cv::rectangle( input_img, temp_rect, Scalar(255,255,255), 2);
			vector<Rect> results;
			vector<double> confs;
			face_det.detectMultiScale(input_img, results, confs, Size(160,160), Size(500,500), 1.2, 0);

			
			for (unsigned long i=0;i<results.size();i++)
			{
				//cv::rectangle( input_img, results[i], Scalar(0,0,0), 2);
				if ( isSameTarget(results[i], temp_rect) > 0.5)
				{
					temp_rect = results[i];
					cout<<"Matched"<<endl;
					matched = true;
					break;
				}
			}

			if(!matched)
				continue;

			//imshow( "show", input_img);
			//waitKey(0);
			
			rects_in_this_image.push_back( temp_rect);
			//for( TiXmlAttribute *iterRect = iterBox->FirstAttribute(); iterRect; iterRect = iterRect->Next())
			//{
			//	// eg: top=90, left=194, width = 37, height =40 
			//	// cout<<iterRect->Name()<<" = "<<iterRect->Value()<<" "; // box contents
			//}
			shape_type temp_shape = shape_type::zeros();
			long counter=0;
			TiXmlElement *iterPart = iterBox->FirstChildElement();
			for (; iterPart; iterPart = iterPart->NextSiblingElement())
			{
				TiXmlAttribute *iterPoint = iterPart->FirstAttribute(); iterPoint = iterPoint->Next();
				temp_shape(counter++,0) = t_scale*atoi(iterPoint->Value());iterPoint = iterPoint->Next();
				temp_shape(counter++,0) = t_scale*atoi(iterPoint->Value());iterPoint = iterPoint->Next();
			}
			shapes_in_this_image.push_back( temp_shape);
		}
		if(!matched)
			continue;

		imgs.push_back(input_img);
		rects.push_back( rects_in_this_image);
		shapes.push_back( shapes_in_this_image);
	}
}

int main( int argc, char** argv)
{
	/* load face dectector */
	scanner fhog_sc;
	if(!fhog_sc.loadModel("super_pack_lfw.xml"))
	{
		cout<<"Can not load the face detector model"<<endl;
		return 1;
	}

	/* prepare the training parameters */
	string train_file_xml = "helen_trainset_info.xml"; // training file
	string image_root = "F:\\data\\facial_point_data\\download\\helen\\trainset\\";
	vector<Mat> imgs;
	vector<vector<Rect> > rects;	// store the rects;
	vector<vector<shape_type> > shapes;	// store the shapes
	
	load_train_file( fhog_sc, train_file_xml, image_root, 256, imgs, rects, shapes);

	cout<<"imgs size "<<imgs.size()<<endl;
	cout<<"rects size "<<rects.size()<<endl;
	cout<<"shapes size "<<shapes.size()<<endl;

	//for (unsigned long i=0;i<imgs.size();i++)
	//{
	//	Mat draw = shape_predictor::draw_shape( imgs[i], rects[i], shapes[i]);
	//	imshow("show", draw);
	//	waitKey(0);
	//}

	shape_predictor_trainer trainer;
	//trainer.set_oversampling_amount(50);
	//trainer.set_feature_pool_size(500);
	//trainer.set_feature_pool_region_padding(0.2);
	//trainer.set_num_test_splits(30);
	//trainer.set_nu(0.2);
	//trainer.set_oversampling_amount(30);
	//trainer.set_nu(0.05);
	//trainer.set_tree_depth(2);
	trainer.set_cascade_depth(15);

	cout<<"trainer cascade depth is "<<trainer.get_cascade_depth()<<endl;
	cout<<"trainer tree depth is "<<trainer.get_tree_depth()<<endl;
	cout<<"trainer number of tree per cascade level   is "<<trainer.get_num_trees_per_cascade_level()<<endl;
	cout<<"trainer nu   is "<<trainer.get_nu()<<endl;
	cout<<"trainer oversamplingamount is "<<trainer.get_oversampling_amount()<<endl;
	cout<<"trainer feature pool size  is "<<trainer.get_feature_pool_size()<<endl;
	cout<<"trainer number of tree splits  is "<<trainer.get_num_test_splits()<<endl;
	cout<<"trainer lambda is "<<trainer.get_lambda()<<endl;


	shape_predictor sp = trainer.train( imgs, rects, shapes);

	cout<<"Train error is "<<trainer.test_shape_predictor(sp, imgs, rects, shapes)<<endl;

	sp.save_model("model.xml");
	sp.load_model("model.xml");


	/* Test the shape predictor */
	string test_file_path = "helen_testset_info.xml";
	image_root = "F:\\data\\facial_point_data\\download\\helen\\testset\\";
	load_train_file( fhog_sc, test_file_path, image_root,256, imgs, rects, shapes);

	cout<<"Test error is "<<trainer.test_shape_predictor(sp, imgs, rects, shapes)<<endl;

	

	for (unsigned long i=0;i<imgs.size();i++)
	{
		vector<shape_type> p_shapes;
		for (unsigned long j=0;j<rects[i].size();j++)
		{
			TickMeter tk;
			tk.start();
			shape_type pre_shape = sp(imgs[i], rects[i][j]);
			tk.stop();
			cout<<"single time "<<tk.getTimeMilli()<<endl;
			p_shapes.push_back(pre_shape);
		}

		Mat draw = shape_predictor::draw_shape( imgs[i], rects[i], p_shapes);
		imshow("show", draw);
		waitKey(0);

	}

	return 0;
}
