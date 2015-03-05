#include <assert.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "misc.hpp"
using namespace std;
using namespace cv;

/* convert from (xmax, xmin, ymax,ymin) to opencv Rect */
Rect bbsToRect( int xmin, int xmax, int ymin, int ymax )
{
	assert( xmin > 0 && ymin > 0 && ymax > ymin && xmax > xmin);
	return Rect( xmin, ymin, xmax - xmin, ymax - ymin);
}

/*  resize the bounding box, based on the center--> doesn't change the Rect's center */
/*  mind result maybe less than zero ~~*/
Rect resizeBbox( const Rect &inrect, double h_ratio, double w_ratio )
{
	int d_width = inrect.width * w_ratio;
	int d_height = inrect.height * h_ratio;
	int d_x = inrect.x + inrect.width/2 - d_width/2;
	int d_y = inrect.y + inrect.height/2 - d_height/2;
	
	return cv::Rect( d_x, d_y, d_width, d_height );
}

/* crop image according to the rect, region will padded with boudary pixels if needed */
Mat cropImage( const Mat &input_image, const Rect &inrect )
{
	Mat outputImage;

	int top_pad=0;
	int bot_pad=0;
	int left_pad=0;
	int right_pad=0;
	
	if( inrect.x < 0)
		left_pad = std::abs(inrect.x);
	if( inrect.y < 0)
		top_pad = std::abs(inrect.y);
	if( inrect.x + inrect.width > input_image.cols )
		right_pad = inrect.x + inrect.width - input_image.cols;
	if( inrect.y+inrect.height >  input_image.rows)		
		bot_pad = inrect.y+inrect.height - input_image.rows;

	Rect imgRect( 0, 0, input_image.cols, input_image.rows );
	Rect target_region_with_pixel = imgRect & inrect;

	copyMakeBorder( input_image(target_region_with_pixel), outputImage, top_pad,bot_pad,left_pad,right_pad,BORDER_REPLICATE);
	return outputImage;
} 

void sampleRects( int howManyToSample, Size imageSize, Size objectSize, vector<Rect> &outputRects)
{
	assert( imageSize.width > objectSize.width && imageSize.height > objectSize.height );
	/* compute nx, ny */
	double x_width  = objectSize.width*1.0;
	double y_height = objectSize.height*1.0;

	double nx_d = std::sqrt( howManyToSample*x_width/y_height ); int nx =int(std::ceil( nx_d +0.1));
	double ny_d = howManyToSample/nx_d; int ny = int( std::ceil(ny_d +0.1));
	
	int x_step = (imageSize.width - objectSize.width)/ nx;
	int y_step = (imageSize.height - objectSize.height) / ny;
	
	outputRects.reserve( nx*ny );

	for( int i=0;i<nx;i++)
	{
		for(int j=0;j<ny;j++)
		{
			Rect tmp( x_step*i, y_step*j, objectSize.width, objectSize.height );
			outputRects.push_back( tmp);
		}
	}
	outputRects.resize( howManyToSample);
}

Rect resizeToFixedRatio( const Rect &inRect,				/* in : input boudingbox informtion */
					     double w_h_ratio,					/* in : target ratio */
						 int flag)							/* in : respect to width  = 0 
																	respect to height > 0 */
{
	int center_x = inRect.x + inRect.width/2;
	int center_y = inRect.y + inRect.height/2;

	if( flag > 0)	/* respect to height*/
	{
		int target_width = int(inRect.height*w_h_ratio);
		int target_x = center_x - target_width/2;
		return Rect( target_x, inRect.y , target_width, inRect.height );
	}
	else			/* respect to width */
	{
		int target_height = int(inRect.width/ w_h_ratio);
		int target_y = center_y - target_height/2;
		return Rect( inRect.x, target_y, inRect.width, target_height );
	}
}

bool saveMatToFile( string path_name, const Mat & m)
{
    ofstream fs(path_name.c_str());
    if( !fs.is_open())
        return false;
    
    if( m.type() == CV_32FC1)
    {
        for( int c=0;c<m.rows;c++)
        {
            for( int j=0;j<m.cols;j++)
            {
                fs<<m.at<float>(c,j)<<" ";
            }
            fs<<endl;
        }
    }
    else if(m.type() == CV_64FC1)
    {
        for( int c=0;c<m.rows;c++)
        {
            for( int j=0;j<m.cols;j++)
            {
                fs<<m.at<double>(c,j)<<" ";
            }
            fs<<endl;
        }

    }

    fs.close();
} 
