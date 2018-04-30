#ifndef __INTEGRAL_HPP__
#define __INTEGRAL_HPP__
#include <string.h>

template<class T1, class T2>
struct  IIDict
{
    T1  *II[3];
    T1  *II_T[3];
    T2  *II2[3];
};


#define IntegralImage_Value(II, W, x, y)        \
    (((II)+(W)+2)[(y)*((W)+1) + (x)])

#define IntegralImage_TopLeft(II, W) ((II) + (W) + 2)
#define IntegralImage_Line(tlII, W, y) ((tlII) + (y)*((W)+1))
#define IntegralImage_Pitch(W) ((W)+1)

#ifdef ENABLE_NEON
/*
  Integral Image is W+1, H+1
*/
int  integral(unsigned char *img,
              int step, int width, int height, 
              unsigned int *II);
#else
template<class T_IN, class T_OUT>
int  integral(T_IN *img, int step, int width, int height, 
              T_OUT *II)
{
    if ((0==img)||(0==II))
        return -1;

    int II_step = width+1;
    memset(II, 0, (II_step)*sizeof(T_OUT));
    
    T_IN *pSrc = img;
    T_OUT *pDst_pre=II+1;
    T_OUT *pDst = pDst_pre + II_step;
    
    // first line
    T_OUT sum = pSrc[0];
    pDst[-1] = 0;
    pDst[0]  = sum;
    for (int w=1; w<width; w++)
    {
        sum += pSrc[w];
        pDst[w] = sum;
    }

    pSrc += step;
    pDst += II_step;
    pDst_pre += II_step;

    for (int h=1; h<height; h++) 
    {
        sum = pSrc[0];
        pDst[-1] = 0;
        pDst[0] = pDst_pre[0] + sum;

        for (int w=1; w<width; w++)
        {
            sum += pSrc[w];
            pDst[w] = pDst_pre[w] + sum;
        }
        pSrc += step;
        pDst += II_step;
        pDst_pre += II_step;
    }
    return 0;
}
#endif

/*
  Compute the sqared image's II
*/
template<class T_IN, class T_OUT>
int  integral2(T_IN *img, int step, int width, int height, 
               T_OUT *II)
{
    if ((0==img)||(0==II))
        return -1;

    T_IN *pSrc = img;
    T_OUT *pDst = II;
    
    int II_step = width+1;
    memset(pDst, 0, (II_step)*sizeof(T_OUT));    

    pDst += II_step+1;
    // first line 
    pDst[-1] = 0;
    pDst[0]  = pSrc[0]*pSrc[0];
    for (int w=1; w<width; w++)
        pDst[w] = pDst[w-1] + pSrc[w]*pSrc[w];

    pSrc += step;
    pDst += II_step;

    for (int h=1; h<height; h++) 
    {
        T_OUT sum = pSrc[0]*pSrc[0];
        pDst[-1] = 0;
        pDst[0] = pDst[-II_step] + sum;

        for (int w=1; w<width; w++)
        {
            sum += pSrc[w]*pSrc[w];
            pDst[w] = pDst[w-II_step]+ sum;
        }
        pSrc += step;
        pDst += II_step;
    }
    return 0;
}

#endif
