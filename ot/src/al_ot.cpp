#include "al_ot.h"
#include "tracker.hpp"
#include "kcftracker.hpp"
#include "dssttracker.hpp"
#include "stapletracker.hpp"
OT_EXPORTS OTHandle ot_create(int maxSide,
                              int minSide,
                              int maxObjNum,
                              int method)
{
    Tracker *handle = 0;
    if (0==method)
        handle = new StapleTracker(maxSide,
                                   minSide,
                                   maxObjNum);
    else if(1==method)
        handle = new KCFTracker(maxSide,
                                minSide,
                                maxObjNum);
    else
        return 0;
    if (0 != handle->init())
    {
        delete handle;
        return 0;
    }
    return (OTHandle)(handle);
}

OT_EXPORTS int  ot_setImage(OTHandle handle,
                            Image_T *img,float & rate)
{
    Tracker *pH = (Tracker *)(handle);
    return pH->setImage(*img,rate);
}

OT_EXPORTS int  ot_addObject(OTHandle handle,
                             Rect_T *pRect,
                             int cate_id)
{
    Tracker *pH = (Tracker *)(handle);
    return pH->add(*pRect, cate_id);
}

OT_EXPORTS int  ot_update(OTHandle handle)
{
    Tracker *pH = (Tracker *)(handle);
    return pH->update();
}

OT_EXPORTS int  ot_count(OTHandle handle)
{
    Tracker *pH = (Tracker *)(handle);
    return pH->count();
}

OT_EXPORTS int  ot_totalCount(OTHandle handle)
{
    Tracker *pH = (Tracker *)(handle);
    return pH->accumCount();
}

OT_EXPORTS int  ot_object(OTHandle handle,
                          unsigned int idx,
                          Rect_T *pRect,
                          int    *pObjId,
                          int    *pCateId,
                          float  *pConf)
{
    Tracker *pH = (Tracker *)(handle);
    Object_T obj;
    if (0 != pH->object(idx, obj))
        return -1;

    *pRect = obj.roi;
    if(0 != pObjId)
        *pObjId = obj.obj_id;
    if(0 != pCateId)
        *pCateId = obj.cate_id;
    if(0 != pConf)
        *pConf = obj.conf;
    return 0;
}

OT_EXPORTS void ot_destroy(OTHandle *pHandle)
{
    Tracker **ppH = (Tracker **)pHandle;
    if (0 != ppH)
    {
        if (0 != *ppH)
            delete (*ppH);
        *ppH = 0;
    }
}
