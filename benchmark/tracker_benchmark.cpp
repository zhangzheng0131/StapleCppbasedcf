#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "al_ot.h"
#include "log.h"
#include "comdef.h"



int usage()
{
    printf("Usage::");
    printf("\t./demo_video [Paras] img.lst\n");
    printf("Paras::\n");
    printf("\tv: set visualization\n");
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
	bool isVisual = true;
    char opts[] = "hv:";
    char oc;
    int method = 0;
    while((oc = getopt_t(argc, argv, opts)) != -1)
    {
        switch(oc)
        {
        case 'h':
            return usage();
		case 'v':
			isVisual = true;
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
	int totalFrame = 0;
    bool isUpdate = false;
	if (isVisual)
	{
		cv::namedWindow("Tracker", 0);
	}

	//open a file to save results
	std::ofstream myfile;
	myfile.open("tracking_results.tmp");


	while (!fileList.eof())
	{
		std::getline(fileList, line);
		std::vector<std::string> eles = split(line);

		if (eles.size() < 1)
			break;
		frame = cv::imread(eles[0].c_str());
		if (frame.empty())
			break;
		totalFrame += 1;

		beg = timeStamp();
		img.format = IMG_FMT_BGRBGR;
		img.pitch[0] = frame.cols * 3;
		img.width = frame.cols;
		img.height = frame.rows;
		img.nPlane = 1;
		img.data[0] = (unsigned char *)frame.data;
        float rate;
		int i = ot_setImage(handle, &img, rate);

		Rect_T roi{ 0,0,0,0 };
		if (isUpdate == false && eles.size() > 1)
		{

			if (5 == eles.size())
			{
				roi.x = int(atof(eles[1].c_str())/rate);
				roi.y = int(atof(eles[2].c_str())/rate);
				roi.w = int(atof(eles[3].c_str())/rate);
				roi.h = int(atof(eles[4].c_str())/rate);
			}
			else
				roi = getRectFromRotatedBB(eles);
			ot_addObject(handle, &roi, 0);
			isUpdate = true;
            roi.x=(int)(roi.x*rate);
            roi.y=(int)(roi.y*rate);
            roi.w=(int)(roi.w*rate);
            roi.h=(int)(roi.h*rate);

			myfile << roi.x << " " << roi.y << " " << roi.w << " " << roi.h << "\n";
			continue;
		}

		if (isUpdate)
		{
			int count = ot_update(handle);
			for (int i = 0; i < count; i++)
			{

				ot_object(handle, i, &roi, 0, 0, 0);
				cv::rectangle(frame,
					cv::Point(roi.x, roi.y),
					cv::Point(roi.x + roi.w,
						roi.y + roi.h),
					cv::Scalar(0, 255, 255), 2, 8);

			}
		}
        roi.x=(int)(roi.x*rate);
        roi.y=(int)(roi.y*rate);
        roi.w=(int)(roi.w*rate);
        roi.h=(int)(roi.h*rate);
		myfile << roi.x << " " << roi.y << " " << roi.w << " " << roi.h << "\n";


		// Show the FPS
		end = timeStamp();
		char str[50] = { 0 };
		double useTime = (end - beg) / 1000;
		if (isVisual)
		{
			sprintf(str, "Time: %0.2f ms", useTime);
			cv::putText(frame, str, cv::Point(20, 30),
			cv::FONT_HERSHEY_SIMPLEX, 1,
			cv::Scalar(255, 0, 0), 2, 8);
			cv::imshow("Tracker", frame);
			char c = (char)cv::waitKey(10);
			if (27 == c || 'q' == c)
				break;
		}
		total_time += useTime;
    }
	myfile.close();
    ot_destroy(&handle);
	float fps = totalFrame / total_time * 1000;
    printf("total time is %0.2f ms; time: %0.2f ms; FPS: %f \n", total_time, total_time/totalFrame , fps);
	myfile.open("fps.tmp");
	myfile << fps;
	myfile.close();
// 	int a;
// 	scanf("%d", &a);
    return 0;
}
