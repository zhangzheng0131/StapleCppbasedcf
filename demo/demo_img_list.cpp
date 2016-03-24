#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "kcftracker.hpp"
#include "log.h"
#include "comdef.h"


// Tracker Wrapper
typedef struct __tagObj
{
    cv::Rect rect;
    int64_t  id;
    int  cateId;
    int  dieNum;
    int  status;
    float trackTh;
}OBJ_T;

class Tracker_Wrapper
{
public:
    Tracker_Wrapper(int maxSide=640, int maxObj=100)
    {
        m_curId = 0;
        m_tracker=0;
        m_objs=0;
        m_maxObj=maxObj;
        m_scale = -1;
        m_maxSide = maxSide;
    }
    ~Tracker_Wrapper(){}
    
    int init()
    {
        if (m_maxObj < 1)
            return -1;

        m_objs = new OBJ_T[m_maxObj];
        if (0==m_objs)
            return -1;
        for (int i=0; i<m_maxObj; i++)
            m_objs[i].id = -1;

        m_tracker = new KCFTracker*[m_maxObj];
        if(0==m_tracker)
            return -1;
        memset(m_tracker, 0,
               m_maxObj*sizeof(KCFTracker*));
        return 0;
    }

    int setImage(cv::Mat &in)
    {
        m_scale=MAX_T(1.f, MAX_T(in.cols*1.f/m_maxSide,
                                 in.rows*1.f/m_maxSide));
        
        int width  = round(in.cols/m_scale);
        int height = round(in.rows/m_scale);
        cv::resize(in, m_frame, cv::Size(width, height));
        return 0;
    }
    
    int add(cv::Rect &roi, int cateId)
    {
        cv::Rect roi_s;
        roi_s.x = ((int)(roi.x/m_scale))/2*2;
        roi_s.y = ((int)(roi.y/m_scale))/2*2;
        roi_s.width = ((int)(roi.width/m_scale))/2*2;
        roi_s.height = ((int)(roi.height/m_scale))/2*2;
        
        if (roi_s.x<0 || roi.y<0 || 
            roi_s.width<2 || roi_s.height<2 ||
            roi_s.x+roi_s.width>=m_frame.cols || 
            roi_s.y+roi_s.height>=m_frame.rows)
            return 0;
        
        int idx = -1;
        int flag = isAlreadyIn(roi_s, idx, cateId);
        
        if (1 == flag)
            return 0;

        if (2==flag)
        {
            float th = 0;
            cv::Rect res = m_tracker[idx]->trackbydetect(m_frame, th,
                                          roi_s);
            if (th > 0.2)
            {
                m_objs[idx].status = 1;
                m_objs[idx].dieNum = 0;
                m_objs[idx].trackTh = th;
                m_objs[idx].rect = res;
            }
        }
        else
        {
            for (int i=0; i<m_maxObj; i++)
            {
                if (m_objs[i].id < 0)
                {
                    idx = i;
                    break;
                }
            }
            if (-1 == idx)
                return 0;
            
            if (0 != m_tracker[idx])
                delete m_tracker[idx];
            m_tracker[idx] = new KCFTracker(true, true,
                                            true, true);
            m_tracker[idx]->init(roi_s, m_frame);
            m_objs[idx].rect = roi_s;
            
            m_objs[idx].id = ++m_curId;
            m_objs[idx].cateId = cateId;
            m_objs[idx].status = 1;
            m_objs[idx].dieNum = 0;
            m_objs[idx].trackTh = 1;
        }
        return 0;
    }
    
    int track()
    {
        for (int i=0; i<m_maxObj; i++)
        {
            if (0 != m_tracker[i])
            {
#if 0
                float th=0;
                cv::Rect res=m_tracker[i]->update(m_frame,
                                                  th);
                if (th < 0.2f)
                {
                    m_objs[i].id = -1;
                    delete m_tracker[i];
                    m_tracker[i] = 0;
                }
                else
                    m_objs[i].rect = res;
#else
                //judge is the object is valid
                if (m_objs[i].dieNum>=10)
                {
                    m_objs[i].id = -1;
                    delete m_tracker[i];
                    m_tracker[i] = 0;
                    continue;
                }
                // If untracked, only wake by detect
                if (0==m_objs[i].status)
                {
                    m_objs[i].dieNum += 1;
                    continue;
                }
                float th=0;
                cv::Rect res=m_tracker[i]->update(m_frame,
                                                  th);
                m_objs[i].rect = res;
                m_objs[i].trackTh =th;

                if (th < 0.2f)
                {
                    m_objs[i].dieNum += 1;
                    m_objs[i].status = 0;
                }
                else
                {
                    m_objs[i].dieNum = 0;
                    m_objs[i].status = 1;
                }
#endif
            }
        }
        return 0;
    }

    inline int count()
    {
        int count = 0;
        for (int i=0; i<m_maxObj; i++)
        {
            if (m_objs[i].id != -1)
                count ++;
        }
        return count;
    }
    inline int totalCount(){ return m_curId;}

    int object(int idx, OBJ_T &obj)
    {
        int idx_v = 0;
        int idx_r = -1;
        for (int i=0; i<m_maxObj; i++)
        {
            if (m_objs[i].id >= 0)
            {
                if (idx == idx_v)
                {
                    idx_r = i;
                    break;       
                }         
                idx_v ++;
            }
        }

        if (-1 == idx_r)
            return -1;

        obj=m_objs[idx_r];
        obj.rect.x = int(obj.rect.x*m_scale);
        obj.rect.y = int(obj.rect.y*m_scale);
        obj.rect.width = int(obj.rect.width*m_scale);
        obj.rect.height = int(obj.rect.height*m_scale);
        return 0;
    }
    
private:
    int isAlreadyIn(cv::Rect &roi, int &idx, int cateId)
    {
        float size_roi = roi.width*roi.height;
        for (int i=0; i<m_maxObj; i++)
        {
            if (m_objs[i].id >= 0)
            {
                int xmin = MAX_T(roi.x, m_objs[i].rect.x);
                int xmax = MIN_T(roi.x+roi.width,
                                 m_objs[i].rect.x+m_objs[i].rect.width);
                int ymin = MAX_T(roi.y, m_objs[i].rect.y);
                int ymax = MIN_T(roi.y+roi.height,
                                 m_objs[i].rect.y+m_objs[i].rect.height);
                
                if ((xmax <= xmin)|| (ymax <= ymin))
                    continue;

                float size_in  = (xmax-xmin)*(ymax-ymin);
                float size_obj = m_objs[i].rect.width * m_objs[i].rect.height;

                if ((cateId==m_objs[i].cateId)&&(0==m_objs[i].status))
                {
                    idx = i;
                    return 2;
                }
                if ((size_in/size_obj > 0.5) ||
                    (size_in/size_roi > 0.5))
                {
                    idx = i;
                    return 1;
                }
            }         
        }
        return 0;
    }
    
private:
    int m_maxSide;
    int64_t m_curId;
    int m_maxObj;
    KCFTracker **m_tracker;
    OBJ_T *m_objs;
    float m_scale;
    cv::Mat m_frame;
};


int usage()
{
    printf("Usage::");
    printf("\t./demo_video [Paras] img.lst res.txt\n");
    printf("Paras::\n");
    printf("\ts: Save the video result\n");
    printf("\tf: Feature name [lab ,hog]. Default lab\n");
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
    char opts[] = "hsf:";
    char oc;
    std::string feaName="lab";
    int isSave = 0;
    while((oc = getopt_t(argc, argv, opts)) != -1)
    {
        switch(oc)
        {
        case 'h':
            return usage();
        case 'f':
            feaName = getarg_t();
            break;
        case 's':
            isSave = 1;
            break;
        }
    }
    argv += getpos_t();
    argc -= getpos_t();
    
	if (argc<2)
        return usage();

    cv::VideoWriter outputV;
	bool HOG = false;
	bool FIXEDWINDOW = true;
	bool MULTISCALE = true;
	bool LAB = false;

    if (feaName.compare("hog") == 0)
        HOG = true;
    else if (feaName.compare("lab") == 0)
    {
        HOG = true;
        LAB = true;
    }

    Tracker_Wrapper tracker(320, 100);
    if (0!=tracker.init())
        return -1;
    
    std::ifstream fileList(argv[0]);
    if (!fileList.is_open())
        return -1;

    FILE *fOut=fopen(argv[1],"w");
    if (0==fOut)
        return -1;
    
    std::string line, name;
    cv::Mat frame;
    double beg, end, total_time=0;
    int process_num=0;
    while(!fileList.eof())
    {
        std::string line;
        std::getline(fileList, line);
        std::vector<std::string> eles = split(line);

        if (eles.size()<1)
            break;

        frame = cv::imread(eles[0].c_str());
        if (frame.empty())
            break;
        int objNum = (eles.size()-1)/6;

        //save the video if enable
        if (1==isSave && !outputV.isOpened())
        {
            char path[1024] = {0};
            sprintf(path,"./res.avi");
            outputV.open(path,
                         CV_FOURCC('M','J','P','G'),
                         25,
                         cv::Size(frame.cols, frame.rows),
                         true);
            if (!outputV.isOpened())
            {
                fprintf(stderr, "Write video open failed\n");
                return -1;
            }
        }

        beg = timeStamp();
        tracker.setImage(frame);
        tracker.track();
        for (int i=0; i<objNum; i++)
        {
            cv::Rect roi;
            int id = atoi(eles[i*6+5].c_str());
            roi.x = int(atof(eles[i*6+1].c_str()));
            roi.y = int(atof(eles[i*6+2].c_str()));
            roi.width = int(atof(eles[i*6+3].c_str()))-roi.x;
            roi.height = int(atof(eles[i*6+4].c_str()))-roi.y;
            tracker.add(roi, id);
        }

        fprintf(fOut, "%s", eles[0].c_str());
        //get object
        objNum = tracker.count();

        end = timeStamp();
        process_num += 1;
        total_time += ((end-beg)/1000);
        for (int i=0; i<objNum; i++)
        {
            OBJ_T obj;
            tracker.object(i, obj);
            if (1!=obj.status)
                continue;
            fprintf(fOut, ",%d,%d,%d,%d,%d,%d",
                    obj.rect.x, obj.rect.y,
                    obj.rect.width+obj.rect.x-1,
                    obj.rect.height+obj.rect.y-1,
                    obj.cateId, obj.id);
            if (1==isSave)
            {
                char str[50]={0};
                sprintf(str,"%d", obj.id);
                cv::putText(frame, str,
                            cv::Point(obj.rect.x+obj.rect.width/2,
                                      obj.rect.y+obj.rect.height/2), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255,0,0), 1, 8);
            
                cv::ellipse(frame,
                            cv::RotatedRect(cv::Point(obj.rect.x+obj.rect.width/2,
                                                      obj.rect.y+obj.rect.height/2),
                                            cv::Size(obj.rect.width, obj.rect.height), 0),
                            cv::Scalar(255,0,0),2, 8);
            }
        }
        if (1==isSave)
            outputV << frame;
        fprintf(fOut, "\n");
    }
    fclose(fOut);
    printf("Object Num: %d  Time per Frame: %f ms\n",
           tracker.totalCount(), total_time/process_num);
    return 0;
}
