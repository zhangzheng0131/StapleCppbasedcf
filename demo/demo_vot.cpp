#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "al_tracker.h"
#include "log.h"
#include "comdef.h"
#define VOT_RECTANGLE
#include "vot.h"

int usage()
{
    printf("Usage::");
    printf("\t./demo_video [Paras]\n");
    printf("Paras::\n");
    printf("\tm: method mode [0:Staple, 1:KCF]. Default 0\n");
    printf("\th: Print the help information\n");
    return 0;
}


int main(int argc, char* argv[]){

    // Parse the options
    char opts[] = "hm:";
    char oc;
    int method = 0;
    while((oc = getopt_t(argc, argv, opts)) != -1)
    {
        switch(oc)
        {
        case 'h':
            return usage();
        case 'm':
            method = atoi(getarg_t());
            break;
        }
    }
    argv += getpos_t();
    argc -= getpos_t();

    Image_T img;
    memset(&img, 0, sizeof(Image_T));
    OTHandle handle = ot_create(640, 480, 5, method);
    VOT vot;
    cv::Rect initRoi;
    initRoi << vot.region();
    cv::Mat frame = cv::imread(vot.frame());
    img.format = IMG_FMT_BGRBGR;
    img.pitch[0] = frame.cols*3;
    img.width = frame.cols;
    img.height = frame.rows;
    img.nPlane = 1;
    img.data[0] = (unsigned char *)frame.data;
    ot_setImage(handle, &img);

    Rect_T roi_t = {initRoi.x, initRoi.y,
                    initRoi.width, initRoi.height};
    ot_addObject(handle, &roi_t, 0);
    
    while (!vot.end()) {

        std::string imagepath = vot.frame();

        if (imagepath.empty()) break;

        frame = cv::imread(imagepath);
        img.format = IMG_FMT_BGRBGR;
        img.pitch[0] = frame.cols*3;
        img.width = frame.cols;
        img.height = frame.rows;
        img.nPlane = 1;
        img.data[0] = (unsigned char *)frame.data;
        ot_setImage(handle, &img);

        int count = ot_update(handle);
        Rect_T roi;
        if (count > 0) 
            ot_object(handle, 0, &roi, 0, 0, 0);
        else
        {
            roi.x = 0;
            roi.y = 0;
            roi.w = 1;
            roi.h = 1;
        }
        cv::Rect res = cv::Rect(roi.x, roi.y, roi.w, roi.h);
        vot.report(res);
    }
    return 0;
}
