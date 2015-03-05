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

#include <omp.h> 

using namespace std;
using namespace cv;

namespace bf = boost::filesystem;
namespace bl = boost::lambda;

/* extract the center part of the feature( crossponding to the target) , save it as 
 * clomun in output_data(not continuous) */
void makeTrainData( vector<Mat> &in_data, Mat &output_data, Size modelDs, int binSize)
{
    assert( output_data.type() == CV_32F);
    assert( in_data[0].type() == CV_32F && in_data[0].isContinuous());

	int w_in_data = in_data[0].cols;
	int h_in_data = in_data[0].rows;

	int w_f = modelDs.width/binSize;
	int h_f = modelDs.height/binSize;
    
    assert( w_in_data > w_f && h_in_data > h_f );
	for( int c=0;c < in_data.size(); c++)
	{
        float *ptr=(float*)in_data[c].ptr() + (h_in_data - h_f)/2*w_in_data + (w_in_data - w_f)/2;
        for( int j=0;j<h_f;j++)
        {
            float *pp = ptr + j*w_in_data;
            for( int i=0;i<w_f;i++)
            {
                output_data.at<float>(0,c*h_f*w_f + j*w_f+i) = pp[i];    
            }
        }

	}
}


/*  return the number of files in the folder "in_path" */
size_t getNumberOfFilesInDir( string in_path )
{
    bf::path c_path(in_path);   
    if( !bf::exists(c_path))
        return -1;
    if( !bf::is_directory(c_path))
        return -1;

    int cnt = std::count_if(
        bf::directory_iterator( c_path ),
        bf::directory_iterator(),
        bl::bind( static_cast<bool(*)(const bf::path&)>(bf::is_regular_file), 
        bl::bind( &bf::directory_entry::path, bl::_1 ) ) );
    return cnt;
}

/*  sample windows from the giving folder */
bool sampleWins(  
                  const string &imgpath,            // in : path of the image folder
                  const string &groundtruth_path,   // in : path of the ground true folder
                  bool isPositive,                  // in : sample positive or negative
                  vector<Mat> &samples,             // out: sampled windows, maybe agumented by flip
                  cv::Size target_size,             // in : target size
                  cv::Size padded_size,              // in : padded size
                  int number_of_negative_sample     // in : number of negative samples
                )
{
    cout<<"Sampling  ..."<<endl;
    samples.clear();

    /*  sampling positive samples , require both imgpath and groundtruth_path */
    if( isPositive )
    {
        bf::path pos_img_path( imgpath);
        bf::path pos_gt_path(groundtruth_path);
        if( !bf::exists( pos_img_path) || !bf::exists(pos_gt_path))
        {
            cout<<"pos img or gt path does not exist!"<<endl;
            cout<<"check "<<pos_img_path<<"  and "<<pos_gt_path<<endl;
            return false;
        }
        
        /* iterate the folder  */
        bf::directory_iterator end_it;
        vector<string> image_path_vector;
        vector<string> gt_path_vector;

        for( bf::directory_iterator file_iter(pos_img_path); file_iter!=end_it; file_iter++)
        {
            bf::path s = *(file_iter);
            string basename = bf::basename( s );
            string pathname = file_iter->path().string();
            string extname  = bf::extension( s );
            
            if( extname!=".jpg" && extname!=".bmp" && extname!=".png" &&
                    extname!=".JPG" && extname!=".BMP" && extname!=".PNG")
                continue;

            /* check if both groundTruth and image exist */
            bf::path gt_path( groundtruth_path + basename + ".xml");
            if(!bf::exists( gt_path))   // image already exists ..
            {
                continue;
            }

            image_path_vector.push_back( pathname );
            /* read the gt according to the image name */
            gt_path_vector.push_back(groundtruth_path + basename + ".xml");
        }

        int Nthreads = omp_get_max_threads();
        #pragma omp parallel for num_threads(Nthreads) /* openmp -->but no error check in runtime ... */
        for( int i=0;i<image_path_vector.size();i++)
        {
            Mat im = imread( image_path_vector[i]);

            vector<Rect> target_rects;
            FileStorage fst( gt_path_vector[i], FileStorage::READ | FileStorage::FORMAT_XML);
            fst["boxes"]>>target_rects;
            fst.release();

            /*  resize the rect to fixed widht / height ratio, for pedestrain det , is 41/100 for INRIA database */

            for ( int i=0;i<target_rects.size();i++) 
            {
                target_rects[i] = resizeToFixedRatio( target_rects[i], target_size.width*1.0/target_size.height, 1); /* respect to height */
                /* grow it a little bit */
                int modelDsBig_width =  padded_size.width + 8;
                int modelDsBig_height = padded_size.height + 8;

                double w_ratio = modelDsBig_width*1.0/target_size.width;
                double h_ratio = modelDsBig_height*1.0/target_size.height;

                target_rects[i] = resizeBbox( target_rects[i], h_ratio, w_ratio);
                
                /* finally crop the image */
                Mat target_obj = cropImage( im, target_rects[i]);
                cv::resize( target_obj, target_obj, cv::Size(modelDsBig_width, modelDsBig_height), 0, 0, INTER_AREA);
                Mat flipped_target; cv::flip( target_obj, flipped_target, 1 );
                #pragma omp critical
                {
                    samples.push_back( target_obj );
                    samples.push_back( flipped_target);
                }
            }
        }
    }
    else
    {
        bf::path neg_img_path( imgpath);

		if(!bf::exists(neg_img_path))
		{
			cout<<"negative image folder path "<<neg_img_path<<" dose not exist "<<endl;
			return false;
		}
        int number_of_neg_images = 	getNumberOfFilesInDir( imgpath );
        int number_of_sample_per_image = number_of_negative_sample / number_of_neg_images;
        
        vector<string> neg_paths;
        bf::directory_iterator end_it;
        for( bf::directory_iterator file_iter(neg_img_path); file_iter!=end_it; file_iter++)
		{
            string pathname = file_iter->path().string();
			string extname  = bf::extension( *file_iter);
			if( extname!=".jpg" && extname!=".bmp" && extname!=".png" &&
					extname!=".JPG" && extname!=".BMP" && extname!=".PNG")
				continue;
			neg_paths.push_back( pathname );
		}
        
        int Nthreads = omp_get_max_threads();
        #pragma omp parallel for num_threads(Nthreads) /* openmp -->but no error check in runtime ... */
        for ( unsigned int c=0; c<neg_paths.size();c++ ) 
        {
            vector<Rect> target_rects;
            Mat img = imread( neg_paths[c] );
            sampleRects( number_of_sample_per_image, img.size(), target_size, target_rects);
            std::random_shuffle( target_rects.begin(), target_rects.end());
			if( target_rects.size() > number_of_sample_per_image)
			    target_rects.resize( number_of_sample_per_image );
            
            /*  resize and crop the target image */
            for ( int i=0;i<target_rects.size();i++) 
            {
                target_rects[i] = resizeToFixedRatio( target_rects[i], target_size.width*1.0/target_size.height, 1); /* respect to height */
                /* grow it a little bit */
                int modelDsBig_width =  padded_size.width + 8;
                int modelDsBig_height = padded_size.height + 8;

                double w_ratio = modelDsBig_width*1.0/target_size.width;
                double h_ratio = modelDsBig_height*1.0/target_size.height;

                target_rects[i] = resizeBbox( target_rects[i], h_ratio, w_ratio);
                
                /* finally crop the image */
                Mat target_obj = cropImage( img, target_rects[i]);
                cv::resize( target_obj, target_obj, cv::Size(modelDsBig_width, modelDsBig_height), 0, 0, INTER_AREA);
                #pragma omp critical
                {
                    samples.push_back( target_obj );
                }
            }

        }

    }

    cout<<"Sampling done ..."<<endl;
    return true;
}

int main( int argc, char** argv)
{

    /*-----------------------------------------------------------------------------
     *     setting parameters
     *-----------------------------------------------------------------------------*/
    string groundtruth_path = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/GENKI/GENKI-R2009a/opencv_gt/";
    string positive_img_path = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/GENKI/GENKI-R2009a/files/";
    string negative_img_path = "/media/yuanyang/disk1/data/face_detection_database/nature_image_no_face/";

    cv::Size target_size( 80, 80);
    cv::Size padded_size( 96, 96);
    int fhog_binsize = 8;
    int fhog_oritention = 9;
    double neg_pos_numbers_ratio = 1.5;


    /*  start training ... */
    int feature_dim = padded_size.width/fhog_binsize*padded_size.height/fhog_binsize*( 3*fhog_oritention+5); 
    vector<Mat> samples;
    sampleWins( positive_img_path, groundtruth_path, true, samples,  target_size, padded_size, 0);

    /*  creating the positive features, row samples */
    Mat positive_feature = Mat::zeros( samples.size(), feature_dim, CV_32F );
    feature_Pyramids feature_generator;
    for( unsigned int c=0;c<samples.size();c++)
    {
        Mat img = samples[c];
        Mat fhog_feature;
        vector<Mat> f_chns;
        feature_generator.fhog( img, fhog_feature, f_chns, 0, fhog_binsize, fhog_oritention, 0.2);
        Mat stored_feature = positive_feature.row( c );
        makeTrainData(  f_chns , stored_feature ,padded_size, fhog_binsize);
        /*  visualize the feature */
        //Mat draw;
        //vector<Mat> zero_pi_channel( f_chns.begin()+18, f_chns.begin()+27);
        //feature_generator.visualizeHog( zero_pi_channel, draw);
        //cout<<"feature size "<<fhog_feature.size()<<endl;
        //cout<<"feature channels "<<f_chns.size()<<endl;
        //imshow("input", img);
        //imshow("fhog", draw);
        //waitKey(0);
    }
    cout<<"Positive samples created, number of samples is "<<positive_feature.rows<<", feature dim is "<<positive_feature.cols<<endl;
    /*  creating negative samples , round 1, random select windows */
    
    sampleWins( negative_img_path, "", false, samples, target_size, padded_size, positive_feature.rows*neg_pos_numbers_ratio);
    Mat negative_feature = Mat::zeros( positive_feature.rows*neg_pos_numbers_ratio, feature_dim, CV_32F);
    for( unsigned int c=0;c<samples.size();c++)
    {
        Mat img = samples[c];
        Mat fhog_feature;
        vector<Mat> f_chns;
        feature_generator.fhog( img, fhog_feature, f_chns, 0, fhog_binsize, fhog_oritention, 0.2);
        Mat stored_feature = negative_feature.row( c );
        makeTrainData(  f_chns , stored_feature ,padded_size, fhog_binsize);
        /*  visualize the feature */
        //Mat draw;
        //vector<Mat> zero_pi_channel( f_chns.begin()+18, f_chns.begin()+27);
        //feature_generator.visualizeHog( zero_pi_channel, draw);
        //cout<<"feature size "<<fhog_feature.size()<<endl;
        //cout<<"feature channels "<<f_chns.size()<<endl;
        //imshow("input", img);
        //imshow("fhog", draw);
        //waitKey(0);
    }
    cout<<"Negative samples created, number of samples is "<<negative_feature.rows<<", feature dim is "<<negative_feature.cols<<endl;

    opencv_warpper_libsvm svm_classifier;
    svm_node **svm_train_data;
    svm_classifier.fromMatToLibsvmNode( positive_feature, svm_train_data);
    cout<<"Generating data used by libsvm"<<endl;



    return 0;
}
