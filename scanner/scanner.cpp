#include <iostream>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "../chnfeature/Pyramid.h"
#include "scanner.h"

using namespace cv;
using namespace std;

scanner::scanner()
{

}

scanner::~scanner()
{

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
                cout<<"m_weight is "<<m_weight[counter]<<" feature is "<<feature_ptr[x_index]<<" sum is "<<sum_score<<endl;
                sum_score +=feature_ptr[x_index]*m_weight[counter++];   
            }
        }
    }
    cout<<"counter is "<<counter<<endl;
    return sum_score;
}


bool scanner::slide_image( const Mat &input_img,        // in: input image
                           vector<Rect> &results,       //out: output targets' position
                           vector<float> &confidence,   //out: targets' confidence
                           int stride_factor)           //in : step factor, actual step size will be stride_factor*m_fhog_binsize
{

    /*  Compute the fhog feature  */
    vector<Mat> feature_chns;
    m_feature_geneartor.fhog( input_img, m_computed_feature, feature_chns, 0, m_fhog_binsize, m_fhog_orientation, 0.2); // 0 -> fhog, 0.2 -> clip value

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
            int yt;
            cin>>yt;
            if( det_score > 0)
            {
                results.push_back( Rect( x*m_fhog_binsize, y*m_fhog_binsize, m_padded_size.width, m_padded_size.height) );
                confidence.push_back( det_score );
            }
        }
    }
    
}

bool scanner::detectMultiScale( const Mat &input_image,      //in : input image
                                vector<Rect> results,        //out: output targets' position
                                vector<float> confidence,    //out: targets' confidence
                                const Size &minSize,         //in : min target size 
                                const Size &maxSize,         //in : max target size
                                double scale_factor,         //in : factor to scale the image
                                int stride_factor)           //in : step factor, actual step size will be stride_factor*m_fhog_binsize
{
    if( !checkParameter())
        return false;

    
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
    m_feature_dim = (m_weight_vector.rows > m_weight_vector.cols? m_weight_vector.rows: m_weight_vector.cols);

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

    if( m_weight_vector.cols!=1 && m_weight_vector.rows!=1 )
    {
        cout<<"Weight vector should be one column or one row "<<endl;
        return false;
    }
    return true;
}


