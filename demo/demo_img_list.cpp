#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <armadillo>

#include <opencv2/opencv.hpp>
#include "al_ot.h"
#include "log.h"
#include "comdef.h"
#include <stdio.h>
#include <stdlib.h>

//using namespace arma;

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

//int frameW;
//int frameH;

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
    float ox1 = atof(eles[1].c_str());
    float oy1 = atof(eles[2].c_str());
    float ox2 = atof(eles[3].c_str());
    float oy2 = atof(eles[4].c_str());
    float ix1 = atof(eles[5].c_str());
    float iy1 = atof(eles[6].c_str());
    float ix2 = atof(eles[7].c_str());
    float iy2 = atof(eles[8].c_str());

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

int main(int argc, char* argv[]){
    // Parse the options
    char opts[] = "hsm:";
    char oc;
    int method = 0;
    bool isSave = false;
    int numdead=1;
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
        }
    }
    argv += getpos_t();
    argc -= getpos_t();
    
	if (argc<1)
        return usage();
    cv::VideoWriter outputV;
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
    //cv::namedWindow("Tracker1", 1 );
    //cv::imshow("Tracker:",frame);
    while(!fileList.eof())
    {
        std::getline(fileList, line);
        std::vector<std::string> eles = split(line);
        if (eles.size()<1)
            break;

       // mat A= randu<mat>(5,5)*10;
       // A.print("A;\n");

        printf("frame NO.%d: \n",numdead);
        numdead++;
        std::string a="afsd";
       // if(a.compare("afsd")==0)
      //  {
      //      printf("a.compare return 0 when the same string are compared\n");
      //  }
      //  else{
      //      printf("a.compare return 1 when the same string are compared\n");
      //  }
        frame = cv::imread(eles[0].c_str());
       // frameW=frame.cols;
        //frameH=frame.rows;

       // std::vector<cv::Mat> aplit12;
        //cv::split(frame, aplit12);
       // cv::Mat atemp=cv::Mat(aplit12[0].cols,aplit12[0].rows,CV_8U);
       // cv::transpose(aplit12[0],atemp);
       // cv::Mat atemp2=cv::Mat(aplit12[0].cols,aplit12[0].rows,CV_32F);
        //atemp.convertTo(atemp2, CV_32F);
        //arma::mat Amat( reinterpret_cast<double *>(atemp2.data),atemp.rows,atemp.cols);
        //arma::mat Amat( reinterpret_cast<int*>(atemp.data),atemp.rows,atemp.cols);

      // mat A = aplit12[0];
       // A.print("frame[0]:\n");
        //Amat.print("A;\n");
       // printf("zhangzheng:\n");
        int izz=0;

        //output file
        FILE* f = fopen("/tmp/data.txt", "w+");

        //std::vector<cv::Mat> splitfr;
        //cv::split(frame,splitfr);

       /* for(int c=0;c<frame.channels();c++)
            for(int r=0;r<frame.rows;r++)
                for(int j=0;j<frame.cols;j++) {

                    //printf("%d\t", frame.data[izz]);
                    fprintf(f, "%d\n", frame.data[izz]);
                    //printf("%d ", splitfr[c].at<CV_8U>(r, j));
                    izz++;
                }

        fclose(f);*/
       // printf("-----------------end-----------------");

       /* for(int i=0;i<frame.rows;i++)
        {
            for(int j=0;j<frame.cols;j++)
            {
                for(int n=0;n<frame.channels();n++)
                {
                    if(n==0 ) {
                    // printf("%d\t", z.data[i * roiGrayzz.cols + j]);
                      //  printf("%d\t", frame.data[izz]);
                        //printf("%d\t", pzz[0]);
                    }
                    if(j==frame.cols-1) printf("\n");
                    // pzz++;
                    izz++;
                }
            }
        }*/
        if (frame.empty()) {
            printf("cannot access frame\n");
            break;
        }
        //cv::imshow( "Tracker1", frame);
        //cvWaitKey(10);
        // Open saved video
        if (isSave && (!outputV.isOpened()))
        {
            char path[1024] = {0};
            sprintf(path,"%s_res.avi", argv[0]);
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
        Rect_T roi;
        int x,y,w,h;
        float rate;
        ot_setImage(handle, &img, rate);

        //m_maxSide;

        
        if (isUpdate==false && eles.size()>1)
        {
            Rect_T roi;
            if (5==eles.size())
            {
                roi.x = int(atof(eles[1].c_str())/rate);
                roi.y = int(atof(eles[2].c_str())/rate);
                roi.w = int(atof(eles[3].c_str())/rate);
                roi.h = int(atof(eles[4].c_str())/rate);
            }
            else
                roi = getRectFromRotatedBB(eles);
            ot_addObject(handle, &roi,0);
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
                              cv::Scalar(0,255,255),2,8);
            }
        }

     //   isSave=true;
        if (isSave)
            outputV << frame;
        // Show the FPS
        end = timeStamp();
        isSave=false;
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
