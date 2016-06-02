#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "al_tracker.h"
#include "log.h"
#include "comdef.h"

bool isUpdate = false;
bool isSelecting = false;
bool isNewObj    = false;
cv::Rect newObj;
cv::Point origin;

// video stream
cv::VideoCapture g_cap;
FILE *g_img_lst=0;
int  g_file_mode = 0;// 0 video, 1 image list

static void onMouse(int event, int x, int y, int, void* frame)
{
    cv::Mat *image = (cv::Mat *)frame;
    if( isSelecting)
    {
        newObj.x = MIN_T(x, origin.x);
        newObj.y = MIN_T(y, origin.y);
        newObj.width = std::abs(x - origin.x);
        newObj.height = std::abs(y - origin.y);
        newObj &= cv::Rect(0,0,image->cols,image->rows);
    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = cv::Point(x,y);
        newObj = cv::Rect(x,y,0,0);
        isSelecting = true;
        break;
    case CV_EVENT_LBUTTONUP:
        isSelecting = false;
        if( newObj.width > 0 && newObj.height > 0 )
            isNewObj = true;
        break;
    }
}

static int nextFrame(cv::Mat &frame)
{
    cv::Mat tmp;
    if (0==g_file_mode)
    {
        g_cap >> tmp;
        if (tmp.empty())
            return -1;
    }
    else if(1==g_file_mode)
    {
        char path[1024] = {0};
        fgets(path, 1024, g_img_lst);
        if (strlen(path)<4)
            return -1;
        path[strlen(path)-1] = 0;
        tmp = cv::imread(path);
    }

    cv::Size dsize = cv::Size(640,480);
    cv::resize(tmp, frame, dsize);
    return 0;
}

int usage()
{
    printf("Usage::");
    printf("\t./demo_video [Paras] [Video.mp4]\n");
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
    
	if (0==argc) 
        g_cap.open(0);
    else
        g_cap.open(argv[0]);

    if (!g_cap.isOpened())
        return -1;

	// Create Tracker object
    Image_T img;
    memset(&img, 0, sizeof(Image_T));
    OTHandle handle = ot_create(640, 480, 10, method);
    if (0 == handle)
    {
        fprintf(stderr, "Create handle failed\n");
        return -1;
    }
    
    cv::Mat frame, show;
    cv::namedWindow("Tracker", 0 );
    cv::setMouseCallback("Tracker", onMouse, &show);    
    bool paused = false;
    double beg, end;
    while (true)
    {
        if (!paused)
        {
            if (0!= nextFrame(frame))
                break;
        }
        frame.copyTo(show);   
        
        //Do Tracking        
        beg = timeStamp();
        img.format = IMG_FMT_BGRBGR;
        img.pitch[0] = frame.cols*3;
        img.width = frame.cols;
        img.height = frame.rows;
        img.nPlane = 1;
        img.data[0] = (unsigned char *)frame.data;
        
        ot_setImage(handle, &img);
        if (isNewObj)
        {
            Rect_T roi = {newObj.x, newObj.y,
                          newObj.width, newObj.height};
            ot_addObject(handle, &roi, 0);
            isNewObj = false;
            isUpdate = true;
		}
        if (isUpdate)
        {
            float th = 0;
            int count = ot_update(handle);
            for (int i=0; i<count; i++)
            {
                Rect_T roi;
                ot_object(handle, i, &roi, 0, 0, 0);
                cv::rectangle(show,
                              cv::Point(roi.x, roi.y),
                              cv::Point(roi.x+roi.w,
                                        roi.y+roi.h),
                              cv::Scalar(0, 255, 255),1,8);
            }
        }
            
        // Show the FPS
        end = timeStamp();
        char str[50]={0};
        sprintf(str,"Time: %0.2f ms", (end-beg)/1000);
        cv::putText(show, str, cv::Point(20,30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(255,0,0), 2, 8);
        
        //End Tracking
        if (isSelecting || isNewObj)            
            cv::rectangle(show,newObj,cv::Scalar(255,0,0));
        cv::imshow( "Tracker", show);
        char c = (char)cv::waitKey(10);
        if(27==c || 'q'==c)
            break;
        else if ('p'==c || 32==c)
            paused = !paused;
    }
    ot_destroy(&handle);
    return 0;
}
