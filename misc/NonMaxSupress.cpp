#include "NonMaxSupress.h"


template < typename T >
struct IdxElem
{
    int idx;
    T val;
};

template < typename T>
bool operator < (const IdxElem<T> &a, const IdxElem<T> &b)
{
    return a.val < b.val;
}

/*
 * return the sorted result of vals by its indice. That is for each
 * i < j, we have vals[idx[i]] < vals[idx[j]]
 */
template <typename T>
void SortIdx(std::vector<T> &vals, std::vector<int> &idx)
{
    int len = vals.size();
    std::vector< IdxElem<T> > elem;
    elem.resize(len);
    for (int i=0; i<len; i++)
    {
        elem[i].idx = i;
        elem[i].val = vals[i];
    }

    std::sort(elem.begin(), elem.end());

    idx.resize(len);
    for (int i=0; i<len; i++)
    {
        idx[i] = elem[i].idx;
    }
}

void NonMaxSupress(std::vector<cv::Rect> &boxes, std::vector<double> &scores,
                   double overlap_threshold , int type)
{
    assert( boxes.size() == scores.size() );
    assert( overlap_threshold > 0.0 && overlap_threshold < 1.0 );

    int numBox = boxes.size();
    int xx1, yy1, xx2, yy2;
    int idx1, idx2;
    double area, ratio;
    std::vector<int> idx;
    std::vector<cv::Rect> resBoxes;
    std::vector<double> resScores;
    std::vector<double> areas;
    std::vector<bool> supress;  // supress[i] set to TRUE is boxes[i] is supressed
    cv::Rect rect1, rect2;

    // sort with indice
    SortIdx(scores, idx);

    areas.resize(numBox);
    supress.resize(numBox);
    for (int i=0; i<numBox; i++)
    {
        areas[i] = boxes[i].width*boxes[i].height;
        supress[i] = false;
    }

    // from higher score to lower score
    for (int i=numBox-1; i>=0; i--)
    {
        if ( (type&0x0F) == NMS_MAXG && supress[idx[i]] )
            continue;

        idx1 = idx[i];
        rect1 = boxes[idx1];
        if ( !supress[idx1] )
        {
            resBoxes.push_back( rect1 );
            resScores.push_back( scores[idx1]);
        }

        for (int j = i-1; j>=0; j--)
        {
            idx2 = idx[j];
            rect2 = boxes[idx2];

            // compute overlap ratio
            xx1 = std::max(rect1.x, rect2.x);
            yy1 = std::max(rect1.y, rect2.y);
            xx2 = std::min(rect1.x+rect1.width, rect2.x+rect2.width);
            yy2 = std::min(rect1.y+rect1.height, rect2.y+rect2.height);
            if ( xx2 < xx1 || yy2 < yy1 )
                area = 0.0;
            else
                area = (xx2-xx1)*(yy2-yy1);
            if ( (type&0x0F0) == NMS_UNION )
                ratio = area/(areas[idx1]+areas[idx2]-area);
            else
                ratio = area/std::min(areas[idx1],areas[idx2]);

            if ( ratio > overlap_threshold )
            {
                // supress high overlap window
                supress[idx2] = true;
            }
        }
    }

    boxes = resBoxes;
    scores = resScores;
}
