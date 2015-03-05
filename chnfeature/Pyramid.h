#ifndef PYRAMID_H
#define PYRAMID_H

#include <iostream>
#include <fstream> 
#include <cmath>
#include <vector>
#include <typeinfo>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp" 

using namespace std;
using namespace cv;

struct channels_opt 
{
	int nPerOct ;//number of scales per octave
	int nOctUp ; //number of up_sampled octaves to compute
	int shrink;  
	int smooth ;//radius for channel smoothing (using convTri)
	int nbins;  //number of orientation channels
	int binsize;//spatial bin size
	int nApprox;// number of approx
	Size minDS ; //minimum image size for channel computation
	Size pad;
	channels_opt ()
	{
		nPerOct=8 ;
		nOctUp=0 ;
        shrink=4;
		smooth =1;
		minDS=Size(41,100) ;
		pad=Size(12,16);
		nbins=6;
        binsize= shrink;
		nApprox=7;
	}
};
class feature_Pyramids
{
public:

	feature_Pyramids();

	~feature_Pyramids();

	/* 
     * ===  FUNCTION  ======================================================================
     *         Name:  chnsPyramid
     *  Description:  get image feature channels Pyramids for object detection
     * =====================================================================================
     */
	bool chnsPyramid(const Mat &img,                                    //in:  image
                    vector<vector<Mat> > &approxPyramid,			    //out: feature channels pyramid
                    vector<double> &scales,							            //contain:really compute && approx
                    vector<double> &scalesh,						    //out: all scales
                    vector<double> &scalesw) const;					    //out: the height of per layer
																	    //out: the width of per layer
	bool chnsPyramid(const Mat &img,
					 vector<vector<Mat> > &chns_Pyramid,			     //in: image
					 vector<double> &scales							     //out: feature channels pyramid
					 ) const;										     //out: all scales
	 /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  convTri
     *  Description:   convolve one row of I by a 2rx1 triangle filter
     * =====================================================================================
     */
	void convTri( const Mat &src,                                        //in:  inputArray
						Mat &dst,										 //out: outpuTarray
						const Mat &Km									 //in:  the kernel of Convolution
						) const;        
  	/* ===  FUNCTION  ======================================================================
	*         Name:  get_scales
	*  Description:  get the scales of image features pyramid 
	* =====================================================================================
	*/
	void getscales(const Mat &img,                                        //in:  image
				  vector<Size> &ap_size,								  //out: the size of per layer
				  vector<int> &real_scal,								  //out: the ID of layer we really compute
				  vector<double> &scales,								  //out: all scales
				  vector<double> &scalesh,								  //out: the height of per layer
				  vector<double> &scalesw								  //out: the width of per layer
				  ) const;
	/* ===  FUNCTION  ======================================================================
	*         Name:  get_lambdas
	*  Description:  get lambdas---use the parameter to estimate approximated pyramid layer
	* =====================================================================================
	*/
	void get_lambdas(vector<vector<Mat> > &chns_Pyramid,                  //in:  image feature pyramid
					vector<double> &lambdas,							  //out: lambdas
					vector<int> &real_scal,								  //in:  the layer of image pyramid
					vector<double> &scales								  //in:  all scales 
					)const;
	  /* ===  FUNCTION  ======================================================================
	*         Name:  computeChannels
	*  Description:  compute feature channels--contain:L U V & magnitude & the Gradient Hist(the number depends on parameter--nbins)
	* =====================================================================================
	*/
	void computeChannels( const Mat &image,                                //in:  image
						vector<Mat>& channels							   //out: feature channels
						) const;
	 /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  computeGradMag
     *  Description:  compute gradient magnitude and orientation at each location \
     *                for color image this should be the first channel, (L for LUV, B for BGR), and the channels should be continuous in memory 
     * =====================================================================================
     */
	void computeGradient(const Mat &img,                             //in:  image
                         Mat& grad1, 								 //out: Gradient magnitude 0
                         Mat& grad2, 								 //out: Gradient magnitude 1
                         Mat& qangle1,								 //out: Gradient angle 0
                         Mat& qangle2,								 //out: Gradient angle 1
                         Mat& mag_sum_s) const;						 //out: Gradient magnitude
	/* ===  FUNCTION  ======================================================================
	*         Name:  setParas
	*  Description:   set the parameter of this object
	* =====================================================================================
	*/
	void setParas (const  channels_opt  &in_para ) ;
	/* ===  FUNCTION  ======================================================================
	*         Name:  compute_lambdas
	*  Description:  compute lambdas---use the parameter to estimate approximated pyramid layer
	* =====================================================================================
	*/
	bool compute_lambdas(const vector<Mat> &fold);

	/* ===  FUNCTION  ======================================================================
	*         Name:  getParas
	*  Description:  get the parameter of this object
	* =====================================================================================
	*/
	const channels_opt  &getParas() const;




    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  computeChannels_sse
     *  Description:  compute channel features, same effect as computeChannels, only use sse
     *                channels data is continuous in momory, LUVGOG1G2G3G4G5G6 
     * =====================================================================================
     */
    bool computeChannels_sse( const Mat &image,             // in : input image, BGR 
                              vector<Mat>& channels) const; //out : 10 channle features, continuous in memory
    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  convTri
     *  Description:   convolve one row of I by a 2rx1 triangle filter
     * =====================================================================================
     */
    // sse version, faster
	void convTri( const Mat &src,       // in : input data, for color image, this is the first channel, and set dim =3
                  Mat &dst,             // out: output data, for color image, this is the first channel, and set dim =3
                  int conv_size,        // in : value of r, the length of the kernel
                  int dim) const;       // in : dim, DO NOT SET dim=3 for gray image, and should make 3 channels continuous 


    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  convt_2_luv
     *  Description:  convert the image from BGR to LUV, LUV channel is contiunous in memory
     * =====================================================================================
     */

	bool convt_2_luv( const Mat input_image,            // in : input image
					  Mat &L_channel,                   // out: L channel
					  Mat &U_channel,                   // out: U channel
					  Mat &v_channel) const;            // out: V channel 


    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  computeGradMag
     *  Description:  compute gradient magnitude and orientation at each location (uses sse)\
     *                for color image this should be the first channel, (L for LUV, B for BGR), and the channels should be continuous in memory 
     * =====================================================================================
     */
     bool computeGradMag( const Mat &input_image,      //in  : input image (first channel for color image ) 
                          const Mat &input_image2,     //in  : input image channel 2, set empty for gray image
                          const Mat &input_image3,     //in  : input image channel 3, set empty for gray image
                          Mat &mag,                    //out : output mag
                          Mat &ori,                    //out : output ori
                          bool full,                   //in  : ture -> 0-2pi, otherwise 0-pi
                          int channel = 0              //in  : choose specific channel to compute the mag and ori
                       ) const;


     /* 
      * ===  FUNCTION  ======================================================================
      *         Name:  computeGradHist
      *  Description:  compute the Gradient Hist for oritent, output size will be  w/binSize*oritent x h/binSize, alse continuous in momory
      * =====================================================================================
      */
      bool computeGradHist( const Mat &mag,           //in : mag -> size w x h
                            const Mat &ori,           //in : ori -> same size with mag
                            Mat &Ghist,               //out: gradient hist size - > w/binSize*oritent x h/binSize
                            int binSize,              //in : size of bin, degree of aggregatation
                            int oritent,              //in : number of orientations, eg 6;
                            bool full = false         //in : ture->0-2pi, false->0-pi
                            ) const;


      /* 
       * ===  FUNCTION  ======================================================================
       *         Name:  chnsPyramid_sse
       *  Description:  compute channels pyramid without approximation, slower but accurate
       * =====================================================================================
       */
    bool chnsPyramid_sse(const Mat &img,                        //in : input image
                         vector<vector<Mat> > &chns_Pyramid,    //out: output features
                         vector<double> &scales) const;         //out: scale of each pyramid


/* 
     * ===  FUNCTION  ======================================================================
     *         Name:  chnsPyramid_sse
     *  Description:  compute channels pyramid with approximation, fast
     * =====================================================================================
     */
	bool chnsPyramid_sse(const Mat &img,                                    //in:  image
						vector<vector<Mat> > &approxPyramid,			    //out: feature channels pyramid
						vector<double> &scales,							    //out: all scales
						vector<double> &scalesh,						    //out: the height scales
						vector<double> &scalesw) const;					    //out: the width scales

    
    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  fhog
     *  Description:  compute hog or fhog feature of a given image, 
     *                if img.depth()==CV_32F, the value should be in the range [0, 1]
     * =====================================================================================
     */
    bool fhog( const Mat &img,                  //in : input image ( w x h )
               Mat &fhog_feature,               //out: output feature ( w/binSize*(3*oritent+5) x h/binSize for fhog, w/binSize*(4*oritent) x h/binSize for hog)
               vector<Mat> &hog_channels,       //out: share the same memory with fhog_feature, just a wrapper for operation, each channels -> one orientation
               int type = 0,                    //in :  0 -> fhog, otherwise -> hog
               int binSize = 8,                 //in : binSize ,better be 8
               int oritent = 9,                 //in : oritent ,better be 9 
               float clip = 0.2                 //in : clip value, better be 0.2
               ) const;


    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  visualizeHog
     *  Description:  visualize hog feature for debug
     * =====================================================================================
     */
    void visualizeHog(const vector<Mat> &chns,         // in : each chns corresponding to one orientation
                      Mat &glyphImg,                    //out : show
                      int glyphSize=20, 
                      double range=0.5);

  private:
	  channels_opt  m_opt;
	  vector<double>lam;
      Mat m_normPad;        //pad_size = normPad(5)
      Mat m_km;             //pad_size = smooth(1);

};
Mat get_Km(int smooth);
#endif

