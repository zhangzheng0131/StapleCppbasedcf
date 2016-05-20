#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "kcftracker.hpp"
#include "log.h"
#include "comdef.h"
#define VOT_RECTANGLE
#include "vot.h"

int usage()
{
    printf("Usage::");
    printf("\t./demo_video [Paras]\n");
    printf("Paras::\n");
    printf("\tf: Feature name [lab ,hog]. Default lab\n");
    printf("\th: Print the help information\n");
    return 0;
}


int main(int argc, char* argv[]){

    // Parse the options
    char opts[] = "hf:";
    char oc;
    std::string feaName="lab";
    while((oc = getopt_t(argc, argv, opts)) != -1)
    {
        switch(oc)
        {
        case 'h':
            return usage();
        case 'f':
            feaName = getarg_t();
            break;
        }
    }
    argv += getpos_t();
    argc -= getpos_t();
    
	bool HOG = false;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool LAB = false;

    if (feaName.compare("hog") == 0)
        HOG = true;
    else if (feaName.compare("lab") == 0)
    {
        HOG = true;
        LAB = true;
    }
    
	// Create KCFTracker object
    VOT vot;
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    cv::Rect roi;
    roi << vot.region();
    cv::Mat image = cv::imread(vot.frame());
    tracker.init(roi, image);
    
    while (!vot.end()) {

        std::string imagepath = vot.frame();

        if (imagepath.empty()) break;

        cv::Mat image = cv::imread(imagepath);
        float th = 0;
        cv::Rect res = tracker.update(image, th);
        vot.report(res);
    }    
}
