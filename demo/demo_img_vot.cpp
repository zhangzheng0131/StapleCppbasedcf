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
    printf("\t./demo_video [Paras] groundtruth.txt\n");
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

Rect_T getRectFromRotatedBB(std::vector<std::string> &eles)
{
    float ox1 = atof(eles[0].c_str());
    float oy1 = atof(eles[1].c_str());
    float ox2 = atof(eles[2].c_str());
    float oy2 = atof(eles[3].c_str());
    float ix1 = atof(eles[4].c_str());
    float iy1 = atof(eles[5].c_str());
    float ix2 = atof(eles[6].c_str());
    float iy2 = atof(eles[7].c_str());

    float cx = (ox1+ox2+ix1+ix2)/4.f;
    float cy = (oy1+oy2+iy1+iy2)/4.f;
    
    float x1 = MIN_T(MIN_T(MIN_T(ox1, ox2), ix1), ix2);
    float x2 = MAX_T(MAX_T(MAX_T(ox1, ox2), ix1), ix2);
    float y1 = MIN_T(MIN_T(MIN_T(oy1, oy2), iy1), iy2);
    float y2 = MAX_T(MAX_T(MAX_T(oy1, oy2), iy1), iy2);
    
    float A1 = sqrt((ox1-ox2)*(ox1-ox2)+(oy1-oy2)*(oy1-oy2))*sqrt((ix1-ox2)*(ix1-ox2)+(iy1-oy2)*(iy1-oy2));
    float A2 = (x2 - x1) * (y2 - y1);
    float s = sqrt(A1/A2);
    float w = s * (x2 - x1) + 1;
    float h = s * (y2 - y1) + 1;
    Rect_T res = {round(cx-w/2), round(cy-h/2),
                  round(w),round(h)};
    return res;
}

int process(int method, char *ground_path,
            bool isSave)
{
    cv::VideoWriter outputV;
    Image_T img;
    memset(&img, 0, sizeof(Image_T));
    OTHandle handle = ot_create(640, 480, 10, method);
    
    std::string path=ground_path;
    std::string base_dir = path.substr(0,path.find_last_of("/\\"));
    std::string vName = base_dir.substr(base_dir.find_last_of("/\\")+1, base_dir.size());
    std::ifstream fileList(path.c_str());
    if (!fileList.is_open())
        return -1;
    
    std::string line;
    cv::Mat frame;
    double beg, end, total_time=0;
    bool isUpdate = false;

    cv::namedWindow("Tracker", 0 );
    int frame_num=0;
    while(!fileList.eof())
    {
        std::getline(fileList, line);
        std::vector<std::string> eles = split(line);
        if (eles.size()<1)
            break;
        char img_path[1024] = {0};
        sprintf(img_path, "%s/%08d.jpg",base_dir.c_str(),
                ++frame_num);
        frame = cv::imread(img_path);
        if (frame.empty())
            break;

        // Open saved video
        if (isSave && (!outputV.isOpened()))
        {
            char path[1024] = {0};
            sprintf(path,"%s/../%s_res.avi",base_dir.c_str(),
                    vName.c_str());
            outputV.open(path,
                         CV_FOURCC('M','J','P','G'),
                         25,
                         cv::Size(frame.cols, frame.rows),
                         true);
            if (!outputV.isOpened())
            {
                printf("Write video open failed\n");
                return -1;
            }
        }

        beg = timeStamp();   
        img.format = IMG_FMT_BGRBGR;
        img.pitch[0] = frame.cols*3;
        img.width = frame.cols;
        img.height = frame.rows;
        img.nPlane = 1;
        img.data[0] = (unsigned char *)frame.data;
        ot_setImage(handle, &img);

        Rect_T ground_roi;
        if (4==eles.size())
        {
            ground_roi.x = int(atof(eles[0].c_str()));
            ground_roi.y = int(atof(eles[1].c_str()));
            ground_roi.w = int(atof(eles[2].c_str()));
            ground_roi.h = int(atof(eles[3].c_str()));
        }
        else
            ground_roi = getRectFromRotatedBB(eles);
        if (isUpdate==false && eles.size()>1)
        {
            ot_addObject(handle, &ground_roi, 0);
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
                              cv::Scalar(0,0,255),2,8);
                cv::rectangle(frame,
                              cv::Point(ground_roi.x,
                                        ground_roi.y),
                              cv::Point(ground_roi.x+ground_roi.w,
                                        ground_roi.y+ground_roi.h),
                              cv::Scalar(255,255,0),2,8);
            }
        }

        if (isSave)
            outputV << frame;
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

int main(int argc, char* argv[]){
    // Parse the options
    char opts[] = "hsbm:";
    char oc;
    int method = 0;
    bool isSave = false;
    bool isBatch = false;
    while((oc = getopt_t(argc, argv, opts)) != -1)
    {
        switch(oc)
        {
        case 'h':
            return usage();
        case 'm':
            method = atoi(getarg_t());
            break;
        case 's':
            isSave = true;
            break;
        case 'b':
            isBatch = true;
            break;
        }
    }
    argv += getpos_t();
    argc -= getpos_t();
    
	if (argc<1)
        return usage();

    if (isBatch)
    {
        std::ifstream fileList(argv[0]);
        if (!fileList.is_open())
            return -1;
        int vNum = 0;
        std::string line;
        isSave = true;
        while(!fileList.eof())
        {
            std::getline(fileList, line);
            if (line.size()<1)
                continue;
            char img_path[1024] = {0};
            sprintf(img_path, "%s/groundtruth.txt",
                    line.c_str());
            process(method, img_path, isSave);
        }
    }
    else
        process(method, argv[0], isSave);
    return 0;
}
