#include "jitterImage.h"


void jitterImage(const cv::Mat &img,
                 std::vector<cv::Mat> &out,
                 cv::Size jsiz, int maxn, bool flip,
                 int nTrn, double mTrn, int nPhi, double mPhi,
                 cv::Mat scls, int method)
{
    std::vector<JitterParam> params;
    cv::Rect subRect;
    JitterParam param;
    double stepTrn, stepPhi;
    int ind=0, numParam;

    assert(img.rows>=jsiz.height&&img.cols>=jsiz.width);

    // safety check
    if ( nTrn > 1) {
        stepTrn= (mTrn*2.0)/(nTrn-1);
    } else {
        nTrn = 1;
        stepTrn = 1.0;
        mTrn = 0.0;
    }

    // safety check
    if ( nPhi > 1 ) {
        stepPhi = (mPhi*2.0)/(nPhi-1);
    } else {
        nPhi = 1;
        stepPhi = 1.0;
        mPhi = 0.0;
    }

    // safety check
    if ( scls.empty() ) {
        scls = cv::Mat::ones(1, 2, CV_64F);
    }

    // parse parameter
    numParam = nTrn*nTrn*nPhi*scls.rows;
    params.resize(numParam);
    ind = 0;
    for (double dx=-mTrn; dx<=mTrn+0.0001; dx+=stepTrn)
    {
        for (double dy=-mTrn; dy<=mTrn+0.0001; dy+=stepTrn)
        {
            for (double dp=-mPhi; dp<=mPhi+0.0001; dp+=stepPhi)
            {
                for (int i = 0; i<scls.rows; i++)
                {
                    param.dx    = dx;
                    param.dy    = dy;
                    param.phi   = dp;
                    param.sx    = scls.at<double>(i, 0);
                    param.sy    = scls.at<double>(i, 1);
                    params[ind++] = param;
                }
            }
        }
    }

    // sample paramters if maxn<numParam
    cv::Mat indice(numParam, 1, CV_32S);
    for (int i=0; i<numParam; i++) {
        indice.at<int>(i) = i;
    }
    if ( maxn > 0 && maxn < numParam ){
        cv::randShuffle(indice);
    } else {
        maxn = numParam;
    }

    // target rect
    subRect.x = (img.cols-jsiz.width)/2;
    subRect.y = (img.rows-jsiz.height)/2;
    subRect.width = jsiz.width;
    subRect.height = jsiz.height;

    // generate image for each parameter
    out.clear();
    for (int i=0; i<maxn; i++)
    {
        ind = indice.at<int>(i); // sample parameter
        cv::Mat res = _jitterImage(img, params[ind], method);
        res = res(subRect);
        out.push_back(res.clone());
        if ( flip ) {
            cv::flip(res, res, 1);
            out.push_back(res.clone());
        }
    }
}

cv::Mat _jitterImage(const cv::Mat &img, JitterParam param, int method)
{
    cv::Mat affineMat, res;

    // roate
    affineMat = cv::getRotationMatrix2D(cv::Point2f(img.cols/2.0, img.rows/2.0),
                                        param.phi, 1.0);

    // scale
    affineMat.row(0) = affineMat.row(0)*param.sx;
    affineMat.row(1) = affineMat.row(1)*param.sy;

    // transform
    affineMat.at<double>(2) = affineMat.at<double>(2)+param.dx;
    affineMat.at<double>(5) = affineMat.at<double>(5)+param.dy;

    cv::warpAffine(img, res, affineMat, cv::Size(img.cols, img.rows), method,
                   cv::BORDER_REPLICATE);
    return res;
}
