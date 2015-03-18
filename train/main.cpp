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

/* extract the center part of the feature( crossponding to the target) , save it as 
 * clomun in output_data(not continuous) */
void makeTrainData( vector<Mat> &in_data,   // in : input data, a little larger than modelDs
                    Mat &output_data,       // out: "crop and tiled " feature vector extracted from in_data
                    Size modelDs,           // in : Size of target
                    int binSize)            // in : binSize
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
                  cv::Size padded_size,             // in : padded size
                  int number_of_negative_sample     // in : number of negative samples
                )
{
    cout<<"Sampling  ..."<<endl;
    samples.clear();
    
    /*  openmp lock */
    omp_lock_t image_vector_lock;
    omp_init_lock(&image_vector_lock);

    /*  sampling positive samples , require both imgpath and groundtruth_path */
    if( isPositive )
    {
        bf::path pos_img_path( imgpath);
        bf::path pos_gt_path(groundtruth_path);
        if( !bf::exists( pos_img_path) || !bf::exists(pos_gt_path))
        {
            cout<<"pos img or gt path does not exist!"<<endl;
            cout<<"check "<<pos_img_path<<"  and "<<pos_gt_path<<endl;
            omp_destroy_lock( &image_vector_lock);
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
        cout<<"Scanning folders done, now adding cropped samples"<<endl;

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
            vector<Mat> ori_obj;
            vector<Mat> flipped_obj;
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
                Mat flipped_target; 
                cv::flip( target_obj, flipped_target, 1 );

                ori_obj.push_back( target_obj );
                flipped_obj.push_back( flipped_target );
            }

            for( unsigned int c=0;c< ori_obj.size();c++)
            {
                omp_set_lock( &image_vector_lock);
                samples.push_back( ori_obj[c] );
                samples.push_back( flipped_obj[c]);
                omp_unset_lock( &image_vector_lock);
            }
        }
    }
    else
    {
        bf::path neg_img_path( imgpath);

		if(!bf::exists(neg_img_path))
		{
			cout<<"negative image folder path "<<neg_img_path<<" dose not exist "<<endl;
            omp_destroy_lock( &image_vector_lock);
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
        cout<<"Scanning folder done, now adding cropped samples "<<endl;
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
            
            vector<Mat> ori_obj;
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
                ori_obj.push_back( target_obj);
            }

             for( int c=0;c<ori_obj.size();c++)
             {
                 omp_set_lock( &image_vector_lock);
                 samples.push_back( ori_obj[c] );
                 omp_unset_lock( &image_vector_lock);
             }

        }

    }
    omp_destroy_lock( &image_vector_lock);
    cout<<"Sampling done ..."<<endl;
    return true;
}

void miningHardNegativeSample( scanner &fhog_sc,            // in : detector
                               string negative_img_path,    // in : path of negative image
                               Size padded_size,            // in : Size of the training example |  -- > used to resize the sample
                               Size target_size,            // in : Size of the target size      |
                               vector<Mat> &hard_nega,      // out: hard examples we want
                               int number_to_sample)        // in : how many to sample, -1 to keep all
{
    /*  number_of_sample < 0 means keep all hard examples ..., set 30000 to keep memory save */
    if( number_to_sample < 0 )
        number_to_sample = 30000;

    cout<<"Start mining negative samples "<<endl;
    omp_lock_t image_lock;
    omp_init_lock( &image_lock );
    bf::path img_path( negative_img_path);

    /* reserve the space for hard_nega */
    size_t number_of_imgs = getNumberOfFilesInDir( negative_img_path );
    hard_nega.reserve( number_to_sample );

    /*  how many to sample from each sample  */
    int numbers_to_sample_each_img = number_to_sample / number_of_imgs + 10;
    
    if( !bf::exists( img_path) )
    {
        cout<<"Path does not exist "<<endl;
        omp_destroy_lock( &image_lock );
        return;
    }
   
    vector<string> neg_paths;
    bf::directory_iterator end_it;
    for( bf::directory_iterator file_iter(negative_img_path); file_iter!=end_it; file_iter++)
	{
        string pathname = file_iter->path().string();
		string extname  = bf::extension( *file_iter);
		if( extname!=".jpg" && extname!=".bmp" && extname!=".png" &&
			extname!=".JPG" && extname!=".BMP" && extname!=".PNG")
			continue;
		neg_paths.push_back( pathname );
	}
    
    std::random_shuffle( neg_paths.begin(), neg_paths.end() );

    int Nthreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(Nthreads) /* openmp -->but no error check in runtime ... */
    for( unsigned int c=0;c<neg_paths.size(); c++)
    {
        cout<<"scanning image "<<c<<" "<<neg_paths[c]<<endl;
        Mat input_img = imread( neg_paths[c] );
        vector<Rect> det_target;
        vector<double> det_confs;
        /*  set the threshold value low, adding more hard exapmle , boost the performance */
        fhog_sc.detectMultiScale( input_img, det_target, det_confs, Size(40, 40), Size(200,200), 1.2, 1, 0);

        if( det_target.empty() )
        {
            omp_destroy_lock( &image_lock );
            continue;
        }

        /*  sample from each image */
        std::random_shuffle( det_target.begin(), det_target.end() );
        if( det_target.size() > numbers_to_sample_each_img )
            det_target.resize( numbers_to_sample_each_img);

        for ( unsigned int i=0; i<det_target.size();i++ ) 
        {
            /*  resize  the example to to be a little bit larger than padded_size  */
            int modelDsBig_width =  padded_size.width + 8;
            int modelDsBig_height = padded_size.height + 8;

            double w_ratio = modelDsBig_width*1.0/target_size.width;
            double h_ratio = modelDsBig_height*1.0/target_size.height;

            det_target[i] = resizeBbox( det_target[i], h_ratio, w_ratio);
            /* finally crop the image */
            Mat target_obj = cropImage( input_img, det_target[i]);
            cv::resize( target_obj, target_obj, cv::Size(modelDsBig_width, modelDsBig_height), 0, 0, INTER_AREA);
            if( target_obj.cols < 10 || target_obj.rows < 10 )
            {
                cout<<"Wrong"<<endl;
            }
            omp_set_lock( &image_lock );
            hard_nega.push_back( target_obj );
            omp_unset_lock( &image_lock );
        }
       
    }

    std::random_shuffle( hard_nega.begin(), hard_nega.end());
    if( hard_nega.size() > number_to_sample )
        hard_nega.resize( number_to_sample);
    omp_destroy_lock( &image_lock);
}

int main( int argc, char** argv)
{
    /*-----------------------------------------------------------------------------
     *    1 setting parameters
     *-----------------------------------------------------------------------------*/
    string groundtruth_path = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/GENKI/GENKI-R2009a/opencv_gt/";
    string positive_img_path = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/GENKI/GENKI-R2009a/files/";
    string negative_img_path = "/media/yuanyang/disk1/data/face_detection_database/non_face/";

    string test_img_folder = "/media/yuanyang/disk1/data/face_detection_database/other_open_sets/FDDB/renamed_images/";

    cv::Size target_size( 80, 80);
    cv::Size padded_size( 96, 96);
    int fhog_binsize = 8;
    int fhog_oritention = 9;
    double neg_pos_numbers_ratio = 1.5;
    int Nthreads = omp_get_max_threads();

    /*-----------------------------------------------------------------------------
     *  2 prepare the data for training 
     *-----------------------------------------------------------------------------*/
    /*  start training ... */
    int feature_dim = padded_size.width/fhog_binsize*padded_size.height/fhog_binsize*( 3*fhog_oritention+4); 
    vector<Mat> pos_samples;
    sampleWins( positive_img_path, groundtruth_path, true, pos_samples,  target_size, padded_size, 0);

    //for ( int c=0; c<pos_samples.size(); c++) {
    //    imshow("show", pos_samples[c]);
    //    waitKey(0);
    //}

    /*  creating the positive features, row samples */
    Mat positive_feature = Mat::zeros( pos_samples.size(), feature_dim, CV_32F );
    feature_Pyramids feature_generator;
    #pragma omp parallel for num_threads(Nthreads)
    for( unsigned int c=0;c<pos_samples.size();c++)
    {
        Mat img = pos_samples[c];
        Mat fhog_feature;
        vector<Mat> f_chns;
        feature_generator.fhog( img, fhog_feature, f_chns, 0, fhog_binsize, fhog_oritention, 0.2);
        /* remove the "zero" channel in then end*/
        f_chns.resize(f_chns.size()-1);
        Mat stored_feature = positive_feature.row( c );
        makeTrainData(  f_chns , stored_feature ,padded_size, fhog_binsize);
    }
    cout<<"Positive samples created, number of pos_samples is "<<positive_feature.rows<<", feature dim is "<<positive_feature.cols<<endl;
    vector<Mat>().swap(pos_samples);        // clear the memory

    /*  creating negative samples , round 1, random select windows */
    vector<Mat> neg_samples;
    int top_number_of_neg_samples = positive_feature.rows*neg_pos_numbers_ratio;

    sampleWins( negative_img_path, "", false, neg_samples, target_size, padded_size, top_number_of_neg_samples);
    Mat negative_feature = Mat::zeros( neg_samples.size(), feature_dim, CV_32F);

    #pragma omp parallel for num_threads(Nthreads)
    for( unsigned int c=0;c<neg_samples.size();c++)
    {
        Mat img = neg_samples[c];
        Mat fhog_feature;
        vector<Mat> f_chns;
        feature_generator.fhog( img, fhog_feature, f_chns, 0, fhog_binsize, fhog_oritention, 0.2);
        /* remove the "zero" channel in then end*/
        f_chns.resize(f_chns.size()-1);
        Mat stored_feature = negative_feature.row( c );
        makeTrainData(  f_chns , stored_feature ,padded_size, fhog_binsize);
    }
    cout<<"Negative samples created, number of neg_samples is "<<negative_feature.rows<<", feature dim is "<<negative_feature.cols<<endl;
    opencv_warpper_libsvm svm_classifier;
    svm_parameter svm_para = svm_classifier.getSvmParameters();
    svm_para.gamma = 1.0/positive_feature.cols; // 1/number_of_feature
    svm_classifier.setSvmParameters( svm_para );
    svm_classifier.train( positive_feature, negative_feature, "face_svm.model");
    cout<<"Svm training done "<<endl;
    
    
	/*  -------------------------- train error  Round 1 ---------------------------*/
    int number_of_error = 0;
    Mat predicted_value;
    svm_classifier.predict( negative_feature, predicted_value );
    for( int c=0;c<predicted_value.rows;c++)
    {
        if(predicted_value.at<float>(c,0) > 0)
            number_of_error++;

    }
    svm_classifier.predict( positive_feature, predicted_value );
    for( int c=0;c<predicted_value.rows;c++)
    {
        if( predicted_value.at<float>(c,0) < 0)
            number_of_error++;
    }
    cout<<"Train error rate is "<<1.0*number_of_error/(negative_feature.rows+positive_feature.rows)<<endl;

    /* Save the weight vector in opencv format */
    Mat weight_mat = svm_classifier.get_weight_vector();
    FileStorage fs("svm_weight.xml", FileStorage::WRITE);
    fs<<"svm_weight"<<weight_mat;
    fs.release();

    
    /*-----------------------------------------------------------------------------
     *  now we use the trained svm model to slide on negative imgs, any detected result
     *  will be included in the negative data as the "hard example ", and retrain the 
     *  svm model. This will decrease the FP
     *-----------------------------------------------------------------------------*/

    /*  set scanner */
    FileStorage ffs("svm_weight.xml", FileStorage::READ);
    ffs["svm_weight"]>>weight_mat;
    scanner fhog_sc;
    fhog_sc.setParameters( fhog_binsize, fhog_oritention, target_size, padded_size, weight_mat);
    
    /*  adding hard examples */
    vector<Mat> hard_examples;
    miningHardNegativeSample( fhog_sc, negative_img_path, padded_size, target_size,  hard_examples, -1);
    cout<<"Adding "<<hard_examples.size()<<" hard examples "<<endl;
    
    vector<Mat> second_neg_samples;
    if( hard_examples.size() > top_number_of_neg_samples )
    {
        std::random_shuffle( hard_examples.begin(), hard_examples.end() );
        hard_examples.resize( top_number_of_neg_samples );
        second_neg_samples = hard_examples; // opencv use reference for Mat, it's ok to do that
    }
    else
    {
        // shrink the previous negative samples
        std::random_shuffle( neg_samples.begin(), neg_samples.end() );
        neg_samples.resize( top_number_of_neg_samples - hard_examples.size());
        second_neg_samples = hard_examples;
        second_neg_samples.insert( second_neg_samples.end(), neg_samples.begin(), neg_samples.end());
    }
    negative_feature = Mat::zeros( second_neg_samples.size() , feature_dim, CV_32F);

    #pragma omp parallel for num_threads(Nthreads)
    for( unsigned int c=0;c<second_neg_samples.size();c++)
    {
        if( second_neg_samples[c].empty())
        {
            cout<<"hard examples empty "<<endl;
            continue;
        }
        Mat img = second_neg_samples[c];
        Mat fhog_feature;
        vector<Mat> f_chns;
        feature_generator.fhog( img, fhog_feature, f_chns, 0, fhog_binsize, fhog_oritention, 0.2);
        /* remove the "zero" channel in then end*/
        f_chns.resize(f_chns.size()-1);
        Mat stored_feature = negative_feature.row( c );
        makeTrainData(  f_chns , stored_feature ,padded_size, fhog_binsize);
    }

    /*  Train Round 2 */
    cout<<"Negative samples created, number of samples is "<<negative_feature.rows<<", feature dim is "<<negative_feature.cols<<endl;
    cout<<"Positive samples created, number of samples is "<<positive_feature.rows<<", feature dim is "<<positive_feature.cols<<endl;
    svm_classifier.train( positive_feature, negative_feature, "face_svm.model");
    cout<<"Training Round 2 done "<<endl;
    
    /*  save linear weight */
    weight_mat = svm_classifier.get_weight_vector();
    fs.open( "svm_weight_2.xml", FileStorage::WRITE);
    fs<<"svm_weight"<<weight_mat;
    fs.release();
    
    /*  update the parameters( mainly weight_mat) */
    fhog_sc.setParameters( fhog_binsize, fhog_oritention, target_size, padded_size, weight_mat);
    if(fhog_sc.saveModel("scanner.xml", "libsvm_type"))
        cout<<"Model saved"<<endl;
    else
        cout<<"Model save failed "<<endl;
    return 0;
}
