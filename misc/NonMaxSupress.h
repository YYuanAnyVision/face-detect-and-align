#ifndef NONMAXSUPRESS_H
#define NONMAXSUPRESS_H

#include "opencv2/opencv.hpp"

#define NMS_MAX     0x00
#define NMS_MAXG    0x01
#define NMS_UNION   0x000
#define NMS_MIN     0x010

/*
 * Non-Maximum Supress
 * INPUT
 *  boxes               - bounding box
 *  scores              - score for each box
 *  overlap_treshold    - when multiple box with overlap ratio higer than
 *                      overlap_threshold, keep the one with highest score
 *  tpye                - default is NMS_MAX, NMS_MAXG is greedy verison of
 *                      NMS_MAX, which supressed box doesn't supress over box.
 *                      the 'union' in overlap formula is replaced with 'min' if
 *                      NMS_MIN is set.
 *
 * OUTPU
 *  boxes               - supress result
 */
void NonMaxSupress(std::vector<cv::Rect> &boxes, std::vector<double> &scores,
                   double overlap_threshold = 0.65, int type=NMS_MAXG|NMS_MIN);

#endif // NONMAXSUPRESS_H
