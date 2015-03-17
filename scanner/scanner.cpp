#include <iostream>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "../chnfeature/Pyramid.h"
#include "../misc/NonMaxSupress.h"

#include "scanner.h"

using namespace cv;
using namespace std;

scanner::scanner()
{

}

scanner::~scanner()
{

}


bool scanner::visualizeDetector()
{
    if(!checkParameter())
        return false;
    
    /*  the struct of the fhog feature */
    /* 
     *    2*m_fhog_orientation  --> sensitive orientation channel, 0-2pi
     *    m_fhog_orientation    --> unsensitive orientation channel, 0-pi
     *    4 texture channel
     *    1 zero pad
     *
     *    extract the unsensitive orientation channel and show it
     * */
     vector<Mat> unsentivechannels;
     const float *weight_data = (const float*)(m_weight_vector.data);

     for( unsigned int c=0;c<m_fhog_orientation;c++)
     {
        Mat tmp_channel = Mat::zeros( m_padded_size.height/m_fhog_binsize, m_padded_size.width/m_fhog_binsize, CV_32F);
        /*  the weigth on the same orientation bin */
        const float *weight_pointer1 = weight_data+(c+2*m_fhog_orientation)*( tmp_channel.rows*tmp_channel.cols);
        const float *weight_pointer2 = weight_data+(c+m_fhog_orientation)*( tmp_channel.rows*tmp_channel.cols);
        const float *weight_pointer3 = weight_data+(c)*( tmp_channel.rows*tmp_channel.cols);

        for( int r_index=0;r_index<tmp_channel.rows;r_index++)
        {
            for( int c_index=0;c_index<tmp_channel.cols;c_index++)
            {
                float weight_value = *(weight_pointer1++)+*(weight_pointer2++)+*(weight_pointer3++);
                tmp_channel.at<float>(r_index, c_index) =( weight_value>0?weight_value:0);
            }
        }
        unsentivechannels.push_back( tmp_channel );
     }
     Mat draw;
     m_feature_geneartor.visualizeHog(unsentivechannels, draw, 20, 0.4);
     imshow("detector", draw);
     waitKey(0);
     return true;
}
 

float scanner::get_score( const vector<Mat> &feature_chns,       // in : input feature channels
                          const int &x,                          // in : position in x direction
                          const int &y,                          // in : position in y direction
                          const int &slide_width,                // in : slide target's width in feature map
                          const int &slide_height)               // in : slide target's height in feature map
{
    /*  useful constant */
    const int feature_width = feature_chns[0].cols;
    const int feature_height = feature_chns[0].rows;

    float sum_score = 0.0f;
    long counter = 0;
    for( unsigned int channel_index = 0;channel_index<feature_chns.size();channel_index++)
    {
        const float *channel_ptr = (const float*)( feature_chns[channel_index].data);
        for( unsigned int y_index=0;y_index<slide_height;y_index++)
        {
            const float* feature_ptr = channel_ptr + (y+y_index)*(feature_width) + x;
            for( unsigned int x_index=0;x_index<slide_width;x_index++)
            {
                sum_score +=feature_ptr[x_index]*m_weight[counter++];   
            }
        }
    }
    sum_score += m_weight[m_feature_dim];       // plus the bias term
    return sum_score;
}


bool scanner::slide_image( const Mat &input_img,        // in: input image
                           vector<Rect> &results,       //out: output targets' position
                           vector<double> &confidence,   //out: targets' confidence
                           int stride_factor)           //in : step factor, actual step size will be stride_factor*m_fhog_binsize
{
    /*  Compute the fhog feature  */
    vector<Mat> feature_chns;
    Mat computed_feature;
    m_feature_geneartor.fhog( input_img, computed_feature, feature_chns, 0, m_fhog_binsize, m_fhog_orientation, 0.2); // 0 -> fhog, 0.2 -> clip value
    /* remove the all zero channel */
    feature_chns.resize( feature_chns.size()-1);

    /*  compute useful constant  中国好注释*/
    /*      
     | ------ feature_width ------- |
     --------------x----------------     -----
    |   |- slide_width--|           |      |
    |    ---------------  --        |      |
    |   |               |           |      |
 1  y   |            slide_height   |  feature_heigth 
    |   |               |           |      |
    |    ---------------  --        |      |
    |                               |      |
     -------------------------------     -----
    |                               |
    |    ---------------            |      
    |   |               |           |
 2  |   |               |           |
    |   |               |           |
    |    ---------------            |
    |                               |
     -------------------------------
            .
            .
            .
     -------------------------------
    |                               |
    |    ---------------            |      
    |   |               |           |
 n  |   |               |           |
    |   |               |           |
    |    ---------------            |
    |                               |
     -------------------------------
    */
    const int number_of_channels = feature_chns.size();         // equals n in above figure
    /*  see figure above */
    const int feature_width  = feature_chns[0].cols;
    const int feature_heigth = feature_chns[0].rows;
    const int slide_width  = m_padded_size.width/m_fhog_binsize;
    const int slide_height = m_padded_size.height/m_fhog_binsize;

    for( unsigned int x=0;x<=feature_width-slide_width;x = x+stride_factor)
    {
        for( unsigned int y=0;y<feature_heigth-slide_height;y = y+stride_factor)
        {
            /* compute the score in posision(x,y) */
            float det_score = get_score( feature_chns, x, y, slide_width, slide_height);
            if( det_score > 0)
            {
                results.push_back( Rect( x*m_fhog_binsize, y*m_fhog_binsize, m_padded_size.width, m_padded_size.height) );
                confidence.push_back( det_score );
            }
        }
    }
    return true;
}

bool scanner::detectMultiScale( const Mat &input_image,      //in : input image
                                vector<Rect> &results,       //out: output targets' position
                                vector<double> &confidence,  //out: targets' confidence
                                const Size &minSize,         //in : min target size 
                                const Size &maxSize,         //in : max target size
                                double scale_factor,         //in : factor to scale the image
                                int stride_factor,           //in : step factor, actual step size will be stride_factor*m_fhog_binsize
                                double threshold)            //in : threshold of detection
{
    if( !checkParameter())
        return false;
    /*  compute the scales we will work on  */
    vector<double> scale_vec;
    get_scale_vector( input_image.size(),  minSize ,maxSize, scale_factor, scale_vec);

    /* detect target in each scale */
    Mat processing_image;
    results.clear();
    confidence.clear();
    for( unsigned int scale_index=0;scale_index<scale_vec.size();scale_index++)
    {
        vector<Rect> det_results;
        vector<double> det_confs;
        resize( input_image, processing_image, Size(0,0), scale_vec[scale_index], scale_vec[scale_index]);

        /*  compute stride in each scale */
        int s_stride = int(stride_factor*scale_vec[scale_index]);
        s_stride = s_stride < 1?1:s_stride;
        slide_image( processing_image, det_results, det_confs, s_stride );
        
        for( unsigned int c=0;c<det_results.size();c++)
        {
            if( det_confs[c] < threshold)
                continue;
            Rect tmp_target = det_results[c];
            tmp_target.x /= scale_vec[scale_index];
            tmp_target.y /= scale_vec[scale_index];
            tmp_target.width /= scale_vec[scale_index];
            tmp_target.height /= scale_vec[scale_index];
            
            /*  sometimes the detected result will be slightly out of the image , crop it */
            if( tmp_target.x < 0)
                tmp_target.x = 0;
            if( tmp_target.y < 0)
                tmp_target.y = 0;
            if( tmp_target.x + tmp_target.width > input_image.cols)
                tmp_target.width = input_image.cols - tmp_target.x -1;
            if( tmp_target.y + tmp_target.height > input_image.rows)
                tmp_target.height = input_image.rows - tmp_target.y - 1;

            results.push_back( tmp_target);
            confidence.push_back( det_confs[c] );
        }
    }
    
    /*  group the detected results */
    NonMaxSupress( results, confidence );
    return true;
}

bool scanner::get_scale_vector( const Size &img_size,           // in : used to define the min_Scale
                                const Size &minSize,            // in : minSize
                                const Size &maxSize,            // in : maxSize
                                double scale_factor,            // in : scale factor
                                vector<double> &scale_vec       // out: scale vector
                              ) const 
{
    if( minSize.width>maxSize.width || minSize.height > maxSize.height )
    {
        cout<<"Make sure maxSize > minSize "<<endl;
        return false;
    }
    if( scale_factor < 1 )
    {
        cout<<"Scale_factor should larger than 1 "<<endl;
        return false;
    }

    /*  use height to compute the scale vector, "height" is more robust in image than "width" */
    double max_scale = m_target_size.height*1.0 / minSize.height;
    double min_scale = m_target_size.height*1.0 / maxSize.height;

    /*  also we should keep the image is larger than the slide_window(size m_padded_size) */
    double img_min_scale = max( 1.0*m_padded_size.width/ img_size.width, 1.0*m_padded_size.height/ img_size.height);
    
    double current_scale = max( min_scale, img_min_scale) ;
    while( current_scale < max_scale  )
    {
        scale_vec.push_back( current_scale );
        current_scale *= scale_factor;
    }

    return true; 
}

bool scanner::setParameters( const int &fhog_binsize,        //in : fhog's binsize       
                             const int &fhog_orientation,    //in : fhog's number of orientation
                             const Size &target_size,        //in : target's size
                             const Size &padded_size,        //in : target's padded size( used to scan the image)
                             const Mat &weight_vector)       //in : svm's weight vector
{
    m_fhog_binsize = fhog_binsize;
    m_fhog_orientation = fhog_orientation;

    m_target_size = target_size;
    m_padded_size = padded_size;

    weight_vector.copyTo( m_weight_vector);

    /*  Set the pointer the the data */
    if( weight_vector.type() != CV_32F)
        return false;
    m_weight = (float*)( m_weight_vector.data);
    /*  dimension of m_weight_vector is feature_dim + 1 ( bias term is not counted as feature_dim ) */
    m_feature_dim = (m_weight_vector.rows > m_weight_vector.cols? m_weight_vector.rows: m_weight_vector.cols) - 1;

    /*  check the parameters */
    if(!checkParameter())
    {
        m_weight = NULL;
        return false;
    }
    return true;
}

bool scanner::checkParameter() const
{
    if( m_fhog_binsize <=0 || m_fhog_orientation <=0 )
    {
        cout<<"Check the binsize and orientation"<<endl;
        return false;
    }

    if( m_padded_size.width < m_target_size.width ||
            m_padded_size.height < m_target_size.height ||
            m_target_size.width <=0 || m_target_size.height <=0)
    {
        cout<<"Check the target_size and padded_size "<<endl;
        return false;
    }

    if( m_weight_vector.empty() || !m_weight_vector.isContinuous() || m_weight_vector.type()!=CV_32F)
    {
        cout<<"Weight vector should be continuous and format shoule be CV_32F"<<endl;
        return false;
    }

    if( !m_weight )
    {
        cout<<"m_weight is not set "<<endl;
        return false;
    }

    if( m_weight_vector.cols!=1 && m_weight_vector.rows!=1 )
    {
        cout<<"Weight vector should be one column or one row "<<endl;
        return false;
    }
    return true;
}



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  saveModel
 *  Description:  save the Model
 * =====================================================================================
 */
bool scanner::saveModel( const string &path_to_save,    // in : path
                         const string &infos ) const    // in : informations
{
    if(!checkParameter())
    {
        cout<<"Model is invalid .."<<endl;
        return false;
    }

    FileStorage fs( path_to_save, FileStorage::WRITE);
    if( !fs.isOpened())
    {
        cout<<"Can not open the file "<<path_to_save<<endl;
        return false;
    }
    
    fs<<"fhog_binsize"<<m_fhog_binsize;
    fs<<"fhog_orientation"<<m_fhog_orientation;
    fs<<"target_size"<<m_target_size;
    fs<<"padded_size"<<m_padded_size;
    fs<<"infos"<<infos;
    fs<<"weight_vector"<<m_weight_vector;
    
    return true;
}


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  loadModel
 *  Description:  load the model
 * =====================================================================================
 */
bool scanner::loadModel( const string &path_to_load)        // in : path
{
    FileStorage fs( path_to_load, FileStorage::READ);
    if( !fs.isOpened())
    {
        cout<<"Can not open file "<<path_to_load<<endl;
        return true;
    }
    
    Mat weight_vector;
    Size padded_size;
    Size target_size;
    int feature_dim;
    int fhog_binsize;
    int fhog_orientation;

    fs["weight_vector"]>>weight_vector;
    fs["padded_size"]>>padded_size;
    fs["fhog_binsize"]>>fhog_binsize;
    fs["fhog_orientation"]>>fhog_orientation;
    fs["target_size"]>>target_size;
    fs["infos"]>>m_info;

    setParameters( fhog_binsize, fhog_orientation, target_size, padded_size, weight_vector);
    if( !checkParameter())
    {
        cout<<"Parameters wrong "<<endl;
        return false;
    }
    
    return true;
}
