#ifndef __FFTTOOLS_HPP__
#define __FFTTOOLS_HPP__

//NOTE: FFTW support is still shaky, disabled for now.
/*#ifdef USE_FFTW
#include <fftw3.h>
#endif*/

#include <opencv2/opencv.hpp>
namespace FFTTools
{
    // Previous declarations, to avoid warnings
    cv::Mat fftd(cv::Mat img, bool backwards = false);
	cv::Mat* matCircshift(cv::Mat* mat,int rowMove,int colMove);
    CvMat* matCircshift(CvMat* mat,int rowMove,int colMove);
    cv::Mat real(cv::Mat img);
    cv::Mat imag(cv::Mat img);
    cv::Mat magnitude(cv::Mat img);
    cv::Mat complexMultiplication(cv::Mat a, cv::Mat b);
    cv::Mat complexConjMult(cv::Mat a, cv::Mat b);
    cv::Mat complexSelfConjMult(cv::Mat a);
    cv::Mat complexDivision(cv::Mat a, cv::Mat b);
    cv::Mat complexDivReal(cv::Mat cMat, cv::Mat rMat);
    void rearrange(cv::Mat &img);
    void normalizedLogTransform(cv::Mat &img);
}
#endif
