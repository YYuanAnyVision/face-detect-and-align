#ifndef JITTERIMAGE_H
#define JITTERIMAGE_H

#include "opencv2/opencv.hpp"

struct JitterParam
{
    double dx;  // offset x
    double dy;  // offset y
    double sx;  // scaling x
    double sy;  // scaling y
    double phi; // rotation degree
};

/*
 * INPUTS
 *  img     - image
 *  jsiz    - Final size of of image crop from center
 *  maxn    - maximun number of output. maxn*2 if flip is set
 *  flip    - add reflection of each image
 *  nTrn    - number of translations
 *  mTrn    - max value for translations
 *  nPhi    - number of rotations
 *  mPhi    - max value for rotation ( degree )
 *  scls    - [n,2] Mat, each is horiz/vert scalings
 *  method  - interpolation method for cv::warpAffine
 *
 * OUTPUTS
 *  out     - vector of image
 *
 */
void jitterImage(const cv::Mat &img,
                 std::vector<cv::Mat> &out,
                 cv::Size jsiz,
                 int maxn = -1,
                 bool flip = false,
                 int nTrn = 0,
                 double mTrn = 0.0,
                 int nPhi = 0,
                 double mPhi = 0.0,
                 cv::Mat scls = cv::Mat(),
                 int method = cv::INTER_LINEAR
                 );


/*
 * INPUTS
 *  img     - input image
 *  param   - jitter parametes, see JitterParam for detail
 *  method  - interpolation method for cv::warpAffine
 */
cv::Mat _jitterImage(const cv::Mat &img, JitterParam param,
                     int method= cv::INTER_LINEAR);


#endif // JITTERIMAGE_H
