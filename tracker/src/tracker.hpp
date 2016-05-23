#ifndef  __TRACKER_HPP__
#define  __TRACKER_HPP__

#include "comdef.h"
#include "cimage.hpp"

typedef struct _Object
{
    Rect_T  roi;
    int64_t obj_id;
    int64_t cate_id;
    
    /*
      status: 0 die 1:active -1:invalid
    */
    int     status;
    int64_t act_time;
    int64_t die_time;
    float   conf;
}Object_T;

class Tracker: public CImage_T
{
public:
    Tracker(int maxSide, int minSide, int maxObjNum);
    virtual ~Tracker();
    
public:    
    virtual int init();
    virtual int add(Rect_T &roi, int cate_id=0)=0;
    virtual int update()=0;    
    int object(int idx, Object_T &obj);

    inline int count(){return m_curObjNum;}
    inline int accumCount(){return m_accumObjId;}
public:
    /*
      Update the object or delete an object
    */
    int set(int idx, Rect_T &rect);    
    int reset(int idx);

protected:
    int isAlreadyIn(Rect_T &roi);
    int getValidIdx(int idx);
    int getIdleIdx();
    
protected:
    int   m_maxObjNum;
    int   m_curObjNum;
    int   m_accumObjId;    
    Object_T  *m_objs;
};


#endif

