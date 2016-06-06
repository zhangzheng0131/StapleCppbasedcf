#ifndef __DOLLAR_HPP__
#define __DOLLAR_HPP__

#include <opencv2/core/core.hpp>


cv::Mat fhog(const cv::Mat& input, int binSize = 4,int nOrients = 9,float clip=0.2f,bool crop = false);


#endif
