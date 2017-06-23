#ifndef __AL_OT_H__
#define __AL_OT_H__

#include "comdef.h"

#if defined(__GNUC__)||defined(__clang__)||defined(__llvm__)
#  ifdef SAK_OT_SHARED
#    define OT_EXPORTS __attribute__((visibility("default")))
#  else
#    define OT_EXPORTS 
#  endif
#elif defined(_WINDOWS) || defined(WIN32)
#  ifdef SAK_OT_SHARED
#    define OT_EXPORTS __declspec(dllexport)
#  elif defined(SAK_OT_STATIC)
#    define OT_EXPORTS 
#  else
#    define OT_EXPORTS __declspec(dllexport)
#  endif
#else
#  error "The compiler not supported"
#endif

typedef void* OTHandle;

#ifdef __cplusplus
extern "C" {
#endif
    
    /*
      method
      0: KCF
    */
    OT_EXPORTS OTHandle ot_create(int maxSide,
                                  int minSide,
                                  int maxObjNum,
                                  int method);

    OT_EXPORTS int  ot_setImage(OTHandle handle,
                                Image_T *img, float &);
    OT_EXPORTS int  ot_addObject(OTHandle handle,
                                 Rect_T *pRect,
                                 int cate_id);
    
    OT_EXPORTS int  ot_update(OTHandle handle);

    OT_EXPORTS int  ot_count(OTHandle handle);
    
    OT_EXPORTS int  ot_totalCount(OTHandle handle);
    
    OT_EXPORTS int  ot_object(OTHandle handle,
                              unsigned int idx,
                              Rect_T *pRect,
                              int    *pObjId,
                              int    *pCateId,
                              float  *pConf);
    
    OT_EXPORTS void ot_destroy(OTHandle *pHandle);
    
#ifdef __cplusplus
}
#endif

#endif
