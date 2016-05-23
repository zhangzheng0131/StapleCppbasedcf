#include <math.h>
#include <string.h>
#include "tracker.hpp"

Tracker::Tracker(int maxSide, int minSide, int maxObjNum)
    :CImage_T(maxSide, IMG_FMT_BGRBGR)
{
    m_maxObjNum  = maxObjNum;
    m_curObjNum  = 0;
    m_accumObjId = 0;
    m_objs       = 0;
}

Tracker::~Tracker()
{
    if (0 != m_objs)
        delete []m_objs;
}

int Tracker::init()
{
    if (m_maxObjNum<1)
        return -1;
    
    if (0 != CImage_T::init())
        return -1;
    
    if (0 != m_objs)
        delete []m_objs;
    m_objs = new Object_T[m_maxObjNum];
    if (0 == m_objs)
        return -1;
    memset(m_objs, 0, m_maxObjNum*sizeof(Object_T));
    for(int i=0; i<m_maxObjNum; i++)
    {
        m_objs[i].obj_id = -1;
        m_objs[i].status = -1;
    }
    return 0;
}

int Tracker::object(int idx, Object_T &obj)
{
    int idx_v = getValidIdx(idx);
    if (-1 == idx_v)
        return -1;

    obj = m_objs[idx_v];
    obj.roi.x = round(obj.roi.x*m_scale);
    obj.roi.y = round(obj.roi.y*m_scale);
    obj.roi.w = int(obj.roi.w*m_scale);
    obj.roi.h = int(obj.roi.h*m_scale);
    return 0;
}

int Tracker::set(int idx, Rect_T &rect)
{
    int idx_v = getValidIdx(idx);
    if (-1 == idx_v)
        return -1;
    m_objs[idx_v].roi = rect;
    return 0;
}

int Tracker::reset(int idx)
{
    int idx_v = getValidIdx(idx);
    if (-1 == idx_v)
        return -1;

    memset(m_objs+idx_v, 0, sizeof(Object_T));
    m_objs[idx_v].obj_id = -1;
    m_curObjNum--;
    return 0;
}

////////////////////////
// Common API
////////////////////////
int Tracker::getValidIdx(int idx)
{
    int idx_v = 0;
    int idx_r = -1;
    for (int i=0; i<m_maxObjNum; i++)
    {
        if (m_objs[i].status != -1)
        {
            if (idx == idx_v)
            {
                idx_r = i;
                break;       
            }         
            idx_v ++;
        }
    }
    return idx_r;
}

int Tracker::getIdleIdx()
{
    int idx = -1;
    for (int i=0; i<m_maxObjNum; i++)
    {
        if (-1 == m_objs[i].status)
        {
            idx = i;
            break;
        }
    }
    return idx;
}

int Tracker::isAlreadyIn(Rect_T &roi)
{
    float size_roi = roi.w*roi.h;
    for (int i=0; i<m_maxObjNum; i++)
    {
        if (m_objs[i].status != -1)
        {
            int xmin = MAX_T(roi.x, m_objs[i].roi.x);
            int xmax = MIN_T(roi.x+roi.w,
                             m_objs[i].roi.x+m_objs[i].roi.w);
            int ymin = MAX_T(roi.y, m_objs[i].roi.y);
            int ymax = MIN_T(roi.y+roi.h,
                             m_objs[i].roi.y+m_objs[i].roi.h);
            
            if ((xmax <= xmin)|| (ymax <= ymin))
                continue;

            float size_in  = (xmax-xmin)*(ymax-ymin);
            float size_obj = m_objs[i].roi.w * m_objs[i].roi.h;
            if ((size_in/size_obj > 0.5) ||
                (size_in/size_roi > 0.5))
                return 1;
        }         
    }
    return 0;
}

