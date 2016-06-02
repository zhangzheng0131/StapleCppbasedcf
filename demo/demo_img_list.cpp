#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "al_tracker.h"
#include "log.h"
#include "comdef.h"

int usage()
{
    printf("Usage::");
    printf("\t./demo_video [Paras] img.lst\n");
    printf("Paras::\n");
    printf("\ts: Save the video result\n");
    printf("\tm: method mode [0:Staple, 1:KCF]. Default 0\n");
    printf("\th: Print the help information\n");
    return 0;
}

std::vector<std::string> split(std::string in)
{
    std::vector<std::string> res;
    char *pch=0;
    pch = strtok (&in[0],",");
    while (pch != NULL)
    {
        res.push_back(pch);
        pch = strtok (NULL, ",");
    }
    return res;
}

int main(int argc, char* argv[]){
    // Parse the options
    char opts[] = "hsm:";
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
    
	if (argc<1)
        return usage();
    
    Image_T img;
    memset(&img, 0, sizeof(Image_T));
    OTHandle handle = ot_create(640, 480, 10, method);
    
    std::ifstream fileList(argv[0]);
    if (!fileList.is_open())
        return -1;
    
    std::string line, name;
    cv::Mat frame;
    double beg, end, total_time=0;
    bool isUpdate = false;
    cv::namedWindow("Tracker", 0 );
    while(!fileList.eof())
    {
        std::getline(fileList, line);
        std::vector<std::string> eles = split(line);
        if (eles.size()<1)
            break;
        frame = cv::imread(eles[0].c_str());
        if (frame.empty())
            break;

        //char path[1024]={0};
        //sprintf(path, "%s.BMP", eles[0].c_str());
        //imwrite(path, frame);
        //continue;
        beg = timeStamp();   
        img.format = IMG_FMT_BGRBGR;
        img.pitch[0] = frame.cols*3;
        img.width = frame.cols;
        img.height = frame.rows;
        img.nPlane = 1;
        img.data[0] = (unsigned char *)frame.data;
        ot_setImage(handle, &img);

        if (isUpdate==false && eles.size()>1)
        {
            Rect_T roi = { int(atof(eles[1].c_str())),
                           int(atof(eles[2].c_str())),
                           int(atof(eles[3].c_str())),
                           int(atof(eles[4].c_str()))};
            ot_addObject(handle, &roi, 0);
            isUpdate = true;
            continue;
        }

        if (isUpdate)
        {
            int count = ot_update(handle);
            for (int i=0; i<count; i++)
            {
                Rect_T roi;
                ot_object(handle, i, &roi, 0, 0, 0);
                cv::rectangle(frame,
                              cv::Point(roi.x, roi.y),
                              cv::Point(roi.x+roi.w,
                                        roi.y+roi.h),
                              cv::Scalar(0,255,255),1,8);
            }
        }
        // Show the FPS
        end = timeStamp();   
        char str[50]={0};
        sprintf(str,"Time: %0.2f ms", (end-beg)/1000);
        cv::putText(frame, str, cv::Point(20,30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(255,0,0), 2, 8);
        
        cv::imshow( "Tracker", frame);
        char c = (char)cv::waitKey(10);
        if(27==c || 'q'==c)
            break;
    }
    ot_destroy(&handle);
    return 0;
}
