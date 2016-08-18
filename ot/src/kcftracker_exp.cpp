#include <math.h>
#include "kcftracker_exp.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "dollar_hog.hpp"
#include "labdata.hpp"
#include "comdef.h"
#include "integral.hpp"

static int bgr2gray(cv::Mat& ori, cv::Mat& dst)
{
    dst = cv::Mat(cv::Size(ori.cols, ori.rows),
                  CV_8U);
    unsigned char *ptr = (unsigned char *)(ori.data);
    for (int h=0; h<dst.rows; h++)
    {
        for (int w=0; w<dst.cols; w++)
        {
            uint8_t val =0;
            double tmp = 0;
            tmp = 0.2989*double(ptr[w*3+2])+0.5870*double(ptr[w*3+1]) + 0.1140*double(ptr[w*3]);
            val=int(round(tmp));
            val = MIN_T(val, 255);
            dst.at<uint8_t>(h, w)=val;
        }
        ptr += ori.step[0];
    }
    return 0;
}

static float subPixelPeak(float left, float center, float right)
{   
    float divisor = 2*center - right - left;

    if (divisor == 0)
        return 0;
    return 0.5 * (right-left)/divisor;
}

KCFExpTracker::KCFExpTracker(int maxSide, int minSide,
                       int maxObjNum)
    :Tracker(maxSide, minSide, maxObjNum)
{
    m_trans_cell_size=4;
    m_trans_inner_padding = 0.2f;
    m_trans_fixed_area=150*150;
    m_trans_color_bins = 16;
    m_trans_lr_pwp = 0.04f;
    m_trans_lr_cf = 0.01f;
    m_trans_lambda = 1e-3f;
    m_trans_merge_factor = 0.3f;
    m_trans_y_sigma = 1.f/16;
    //m_trans_y_sigma = 1.f/10;
    m_trans_gauss_kernel_sigma = 0.5f;
    
    //Scale parameter 
    m_scale_num = 33;
    m_scale_cell_size = 4;
    m_scale_max_area = 32*16;
    m_scale_lr = 0.025;
    m_scale_lambda = 1e-3f;
    m_scale_step = 1.02;
    m_scale_y_sigma = 1.f/4;
    m_scale_hann = 0; 
    m_scale_factors = 0; 
}

int KCFExpTracker::init()
{
    if (0!=Tracker::init())
        return -1;

    if (0 != initTransCF())
        return -1;
    
    if (0 != initScaleCF())
        return -1;

    m_cfs.resize(m_maxObjNum);
    return 0;
}

int KCFExpTracker::initTransCF()
{
    return 0;
}

int KCFExpTracker::initScaleCF()
{
    if (0==m_scale_hann)
    {
        m_scale_hann = new float[m_scale_num];
        if (0==m_scale_hann)
            return -1;
        for (int i = 0; i < m_scale_num; i++)
            m_scale_hann[i] = 0.5*(1-std::cos(2*PI_T*i/(m_scale_num-1)));
    }
    
    if (0==m_scale_factors)
    {
        m_scale_factors = new float[m_scale_num];
        for (int i = 0; i < m_scale_num; i++)
            m_scale_factors[i] = pow(m_scale_step,
                                     int(m_scale_num/2)-i);
    }
    
    cv::Mat y = cv::Mat(cv::Size(m_scale_num, 1),
                        CV_32F, cv::Scalar(0));
    
    float sigma = std::sqrt(m_scale_num)*m_scale_y_sigma;
    float mult = -0.5f/(sigma * sigma);
    for (int i = 0; i < m_scale_num; i++)
    {
        int dist = i-int(m_scale_num/2);
        y.at<float>(0, i)=std::exp(mult*dist*dist);
    }
    m_scale_y = FFTTools::fftd(y);
    return 0;
}

int KCFExpTracker::add(Rect_T &roi, int cate_id)
{
    Rect_T roi_s, winR;
    roi_s.x = (int)(roi.x/m_scale);
    roi_s.y = (int)(roi.y/m_scale);
    roi_s.w = ((int)(roi.w/m_scale))/2*2;
    roi_s.h = ((int)(roi.h/m_scale))/2*2;
    
    int idx = getIdleIdx();
    if (-1 == idx)
        return 0;

    if (1 == isAlreadyIn(roi_s))
        return 0;

    float pad = (roi_s.w+roi_s.h)/2.f;
    float bg_w, bg_h, fg_w, fg_h;
    bg_w = round(roi_s.w+pad);
    bg_h = round(roi_s.h+pad);
    float scale = sqrt(m_trans_fixed_area/(bg_w*bg_h));
    bg_w = (int(int(round(bg_w*scale))/m_trans_cell_size)/2*2+1)*m_trans_cell_size/scale;
    bg_h = (int(int(round(bg_h*scale))/m_trans_cell_size)/2*2+1)*m_trans_cell_size/scale;
    fg_w = roi_s.w-pad*m_trans_inner_padding;
    fg_h = roi_s.h-pad*m_trans_inner_padding;

    m_cfs[idx].pos[0] = roi_s.x + roi_s.w/2;
    m_cfs[idx].pos[1] = roi_s.y + roi_s.h/2;
    m_cfs[idx].target_size[0] = roi_s.w;
    m_cfs[idx].target_size[1] = roi_s.h;
    m_cfs[idx].base_tz[0] = roi_s.w;
    m_cfs[idx].base_tz[1] = roi_s.h;        
    m_cfs[idx].bg_size[0] = bg_w;
    m_cfs[idx].bg_size[1] = bg_h;
    m_cfs[idx].fg_size[0] = fg_w;
    m_cfs[idx].fg_size[1] = fg_h;
    m_cfs[idx].scale=scale;
    float radius = MIN_T((bg_w-roi_s.w)*m_cfs[idx].scale,
                         (bg_h-roi_s.h)*m_cfs[idx].scale);
    m_cfs[idx].norm_delta_size = 2*floor(radius/2)+1;
    
    m_objs[idx].roi = roi_s;
    m_objs[idx].cate_id = cate_id;
    m_objs[idx].obj_id = m_accumObjId++;
    m_objs[idx].status = 1;

    //Set the scale parameters
    float factor = 1;
    int norm_tw = round(roi_s.w*m_cfs[idx].scale);
    int norm_th = round(roi_s.h*m_cfs[idx].scale);
    if (norm_tw*norm_th>m_scale_max_area)
        factor = sqrt(m_scale_max_area/(norm_tw*norm_th));
    m_cfs[idx].scale_norm_tz[0] = floor(norm_tw*factor);
    m_cfs[idx].scale_norm_tz[1] = floor(norm_th*factor);
    m_cfs[idx].scale_max_factor = pow(m_scale_step,
                                      floor(log(MIN_T(m_img.width*1.f/roi_s.w, m_img.height*1.f/roi_s.h))/log(m_scale_step)));
    m_cfs[idx].scale_min_factor = pow(m_scale_step,
                                      ceil(log(MAX_T(5.f/bg_w, 5.f/bg_h))/log(m_scale_step)));
    m_cfs[idx].scale_adapt = 1;
    //Train the trackers
    cv::Mat roiImg;
    roiImg = getSubWin(idx);
    
    trainTransCF(idx, roiImg, 1.f, true);
    trainTransPWP(idx, roiImg, 1.f, true);
    trainScaleCF(idx, 1.f, true);
    return 0;
}

int KCFExpTracker::update()
{
    m_curObjNum = 0;
    float train_th = 0.0f;
    float update_th = 0.0f;
    for (int i=0 ;i<m_maxObjNum; i++)
    {
        if (-1 == m_objs[i].status)
            continue;

        float confT;
        detectTrans(i, confT);
        if (confT < update_th)
        {
            printf("trans conf %f\n", confT);
            m_objs[i].status = -1;
            continue;
        }

        float confS;
        detectScaleCF(i, confS);
        if (confS < update_th)
        {
            printf("scale conf %f\n", confS);
            m_objs[i].status = -1;
            continue;
        }

        if (m_cfs[i].pos[0]<0 ||
            m_cfs[i].pos[0]>m_img.width-1 ||
            m_cfs[i].pos[1]<0 ||
            m_cfs[i].pos[1]>m_img.height-1)
        {
            printf("img size: %d, %d pos: %f, %f\n",
                   m_img.width, m_img.height,
                   m_cfs[i].pos[0],
                   m_cfs[i].pos[1]);
            m_objs[i].status = -1;
            continue;
        }
        // get the show rect
        m_objs[i].roi = getShowRect(i);
        
        // Update the trainning model
        if (confT>train_th)
        {
            cv::Mat roiImg;
            roiImg = getSubWin(i);
            trainTransCF(i, roiImg, m_trans_lr_cf, false);
            trainTransPWP(i, roiImg, m_trans_lr_pwp,false);

            if (confS>train_th)
                trainScaleCF(i, m_scale_lr, false);
        }
        m_curObjNum ++;
    }
    return m_curObjNum;
}

Rect_T KCFExpTracker::getShowRect(int idx)
{
    Rect_T roi;
    roi.x = m_cfs[idx].pos[0]-m_cfs[idx].target_size[0]/2;
    roi.y = m_cfs[idx].pos[1]-m_cfs[idx].target_size[1]/2;
    roi.w = m_cfs[idx].target_size[0];
    roi.h = m_cfs[idx].target_size[1];

    roi.x = MAX_T(MIN_T(m_img.width, roi.x), 0);
    roi.y = MAX_T(MIN_T(m_img.height, roi.y), 0);
    roi.w = MIN_T(m_img.width-roi.x-1, roi.w);
    roi.h = MIN_T(m_img.height-roi.y-1, roi.h);
    return roi;
}

int KCFExpTracker::detectTrans(int idx, float &conf)
{
    // Detect by translation CF
    cv::Mat roiImg = getSubWin(idx);
    cv::Mat resCF = detectTransCF(idx, roiImg);
    resCF = cropTransResponseCF(idx, resCF);
    
    // Detect by translation PWP
    roiImg = getSubWinPWP(idx);
    cv::Mat resPWP = detectTransPWP(idx, roiImg);
    resPWP = cropTransResponsePWP(idx, resPWP);
    resCF = 0.7f*resCF + 0.3f*resPWP;

    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(resCF, NULL, &pv, NULL, &pi);
    cv::Point2f pf((float)pi.x, (float)pi.y);
    
#ifdef ENABLE_SUB_PEAK
    if (pi.x>0 && pi.x<resCF.cols-1) {
        pf.x += subPixelPeak(resCF.at<float>(pi.y, pi.x-1),
                             pv,
                             resCF.at<float>(pi.y, pi.x+1));
    }
    if (pi.y>0 && pi.y<resCF.rows-1) {
        pf.y += subPixelPeak(resCF.at<float>(pi.y-1, pi.x),
                             pv,
                             resCF.at<float>(pi.y+1, pi.x));
    }
#endif
    //int center = (m_cfs[idx].norm_delta_size-1)/2;
    pf.x -= resCF.cols/2;
    pf.y -= resCF.rows/2;
    m_cfs[idx].pos[0] += (pf.x)/m_cfs[idx].scale;
    m_cfs[idx].pos[1] += (pf.y)/m_cfs[idx].scale;
    conf = (float)pv;
    return 0;
}

cv::Mat KCFExpTracker::detectTransCF(int idx,
                                  cv::Mat &roiImg)
{
    cv::Mat feaCF;
    getTransFeaCF(roiImg, feaCF);
    applyHann2D(feaCF, m_cfs[idx].hann);
    
    using namespace FFTTools;
    cv::Mat kf = linearCorrelation(feaCF, m_cfs[idx].fea);
    return (real(fftd(complexMultiplication(m_cfs[idx].alpha, kf), true)));
}

cv::Mat KCFExpTracker::detectTransPWP(int idx,
                                   cv::Mat &roiImg)
{
    cv::Mat res = cv::Mat::zeros(roiImg.rows,roiImg.cols,
                                 CV_32F);
    unsigned char *pImg = (unsigned char*)(roiImg.data);
    float *pRes = (float *)(res.data);
    float *pBG = (float *)(m_cfs[idx].bgHist.data);
    float *pFG = (float *)(m_cfs[idx].fgHist.data);
    
    int range = 256/m_trans_color_bins;
    for (int h=0; h<res.rows; h++)
    {
        for (int w=0; w<res.cols; w++)
        {
            int idx1 = pImg[w*3]/range;
            int idx2 = pImg[w*3+1]/range;
            int idx3 = pImg[w*3+2]/range;
            int idx = idx3*m_trans_color_bins*m_trans_color_bins+idx2*m_trans_color_bins+idx1;
            float fg = pFG[idx];
            float bg = pBG[idx];
            if ((fg+bg)<0.000001f)
                *(pRes++) = 0;
            else
                *(pRes++) = fg/(fg+bg);
        }
        pImg += roiImg.step[0];
    }
    return res;
}

cv::Mat KCFExpTracker::getSubWinPWP(int idx)
{
    //Get the search region of PWP
    int norm_pwp_bg_w = round(m_cfs[idx].target_size[0]*m_cfs[idx].scale)+m_cfs[idx].norm_delta_size-1;
    int norm_pwp_bg_h = round(m_cfs[idx].target_size[1]*m_cfs[idx].scale)+m_cfs[idx].norm_delta_size-1;
    int pwp_bg_w = round(norm_pwp_bg_w/m_cfs[idx].scale);
    int pwp_bg_h = round(norm_pwp_bg_h/m_cfs[idx].scale);

    cv::Rect roi;
    float cx = m_cfs[idx].pos[0];
    float cy = m_cfs[idx].pos[1];    
    roi.width = pwp_bg_w;
    roi.height = pwp_bg_h;
    roi.x = round(cx - roi.width/2.f);
    roi.y = round(cy - roi.height/2.f);
    // Get sub Image
    cv::Mat image = cv::Mat(m_img.height,m_img.width,
                            CV_8UC3, m_img.data[0]);
    cv::Mat z = RectTools::subwindow(image, roi,
                                     cv::BORDER_REPLICATE);
    if (z.cols!=norm_pwp_bg_w || z.rows!=norm_pwp_bg_h) 
        cv::resize(z, z, cv::Size(norm_pwp_bg_w,
                                  norm_pwp_bg_h));
    //resize(z,z,cv::Size(norm_pwp_bg_w,norm_pwp_bg_h));
    return z;
}

cv::Mat KCFExpTracker::getSubWin(int idx)
{
    cv::Rect roi;
    float cx = m_cfs[idx].pos[0];
    float cy = m_cfs[idx].pos[1];    
    roi.width = round(m_cfs[idx].bg_size[0]);
    roi.height = round(m_cfs[idx].bg_size[1]);
    roi.x = round(cx - roi.width/2.f);
    roi.y = round(cy - roi.height/2.f);
    int bg_w = round(m_cfs[idx].bg_size[0]*m_cfs[idx].scale);
    int bg_h = round(m_cfs[idx].bg_size[1]*m_cfs[idx].scale);

    // Get sub Image
    cv::Mat image = cv::Mat(m_img.height,m_img.width,
                            CV_8UC3, m_img.data[0]);
    cv::Mat z = RectTools::subwindow(image, roi,
                                     cv::BORDER_REPLICATE);
    if (z.cols != bg_w || z.rows != bg_h) 
        cv::resize(z, z, cv::Size(bg_w, bg_h));
    //resize(z, z, cv::Size(bg_w, bg_h));
    return z;
}

int KCFExpTracker::getTransFeaCF(cv::Mat &roiImg,
                              cv::Mat &feaHog)
{
    //Get HOG Feature
    cv::Mat roiGray;
    bgr2gray(roiImg, roiGray);
    //cv::cvtColor(roiImg, roiGray, CV_BGR2GRAY);
    feaHog = fhog(roiGray, m_trans_cell_size);
    return 0;
}

cv::Mat KCFExpTracker::gaussCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(x1.cols, x1.rows),
                        CV_32F, cv::Scalar(0));
    int N = x1.cols*x1.rows;
    float x1_sum = 0;
    float x2_sum = 0;

    std::vector<cv::Mat> x1_split;
    cv::split(x1, x1_split);
    std::vector<cv::Mat> x2_split;
    cv::split(x2, x2_split);
        
    cv::Mat x1f, x2f, caux;
    for (int i=0; i<x1_split.size(); i++)
    {
        x1f = fftd(x1_split[i]);
        x2f = fftd(x2_split[i]);
        caux = complexConjMult(x1f, x2f); 
        c = c + real(fftd(caux, true));
        x1_sum+=cv::sum(complexSelfConjMult(x1f))[0];
        x2_sum+=cv::sum(complexSelfConjMult(x2f))[0];
    }
    x1_sum /= N;
    x2_sum /= N;
    
    cv::Mat d; 
    cv::max(((x1_sum+x2_sum)-2.f*c)/(x1.cols*x1.rows*x1_split.size()), 0, d);

    cv::Mat k;
    cv::exp((-d/(m_trans_gauss_kernel_sigma*m_trans_gauss_kernel_sigma)), k);
    return fftd(k);
}

cv::Mat KCFExpTracker::linearCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(x1.cols, x1.rows),
                        CV_32FC2, cv::Scalar(0,0));
    
    std::vector<cv::Mat> x1_split;
    cv::split(x1, x1_split);
    std::vector<cv::Mat> x2_split;
    cv::split(x2, x2_split);
        
    cv::Mat caux;
    for (int i=0; i<x1_split.size(); i++)
    {
        c += FFTTools::complexConjMult(fftd(x1_split[i]),
                                       fftd(x2_split[i]));
    }
    c /= (x1.cols*x1.rows*x1.channels());
    return c;
}

int KCFExpTracker::trainTransCF(int idx, cv::Mat &roiImg,
                             float lr, bool isInit)
{
    // Extract the translation feature
    using namespace FFTTools;
    cv::Mat feaCF;
    getTransFeaCF(roiImg, feaCF);
    if (isInit)
    {
        m_cfs[idx].hann = createHann2D(feaCF.rows,
                                       feaCF.cols);
        m_cfs[idx].y=createGaussPeak(idx, feaCF.rows,
                                     feaCF.cols,
                                     m_trans_y_sigma);
    }
    applyHann2D(feaCF, m_cfs[idx].hann);

    // Solving the classifier
#if 1
    cv::Mat kf = real(linearCorrelation(feaCF, feaCF));
    cv::Mat alpha = complexDivReal(m_cfs[idx].y,
                                   kf+m_trans_lambda);
    if (isInit)
    {
        m_cfs[idx].fea = feaCF.clone();
        m_cfs[idx].alpha = alpha.clone();
    }
    else
    {
        m_cfs[idx].fea = (1-lr)*m_cfs[idx].fea + lr*feaCF;
        m_cfs[idx].alpha = (1-lr)*m_cfs[idx].alpha + lr*alpha;
    }
#else
    cv::Mat kf = linearCorrelation(feaCF, feaCF);
    cv::Mat num = complexMultiplication(kf, m_cfs[idx].y);
    cv::Mat den = complexMultiplication(kf,
                                        kf+m_trans_lambda);
    if (isInit)
    {
        m_cfs[idx].fea = feaCF.clone();
        m_cfs[idx].num = num.clone();
        m_cfs[idx].den = den.clone();
    }
    else
    {
        m_cfs[idx].fea = (1-lr)*m_cfs[idx].fea + lr*feaCF;
        m_cfs[idx].num = (1-lr)*m_cfs[idx].num + lr*num;
        m_cfs[idx].den = (1-lr)*m_cfs[idx].den + lr*den;
    }
    m_cfs[idx].alpha=complexDivision(m_cfs[idx].num,
                                     m_cfs[idx].den);
#endif
    return 0;
}

int KCFExpTracker::trainTransPWP(int idx,
                              cv::Mat &roiImg,
                              float lr, bool isInit)
{
    cv::Mat histBg = cv::Mat::zeros(1, m_trans_color_bins*m_trans_color_bins*m_trans_color_bins, CV_32F);
    cv::Mat histFg = cv::Mat::zeros(1, m_trans_color_bins*m_trans_color_bins*m_trans_color_bins, CV_32F);
    
    int bg_h = roiImg.rows, bg_w = roiImg.cols;
    //Get PWP Histgram
    cv::Mat bgMask=cv::Mat::ones(bg_h, bg_w, CV_8U);    
    int offsetX = (m_cfs[idx].bg_size[0]-m_cfs[idx].target_size[0])*m_cfs[idx].scale/2;
    int offsetY = (m_cfs[idx].bg_size[1]-m_cfs[idx].target_size[1])*m_cfs[idx].scale/2;
    bgMask(cv::Rect(offsetX, offsetY,
                    bg_w-2*offsetX, bg_h-2*offsetY))=cv::Scalar::all(0.0);
    
    cv::Mat fgMask=cv::Mat::zeros(bg_h, bg_w, CV_8U);
    offsetX = (m_cfs[idx].bg_size[0]-m_cfs[idx].fg_size[0])*m_cfs[idx].scale/2;
    offsetY = (m_cfs[idx].bg_size[1]-m_cfs[idx].fg_size[1])*m_cfs[idx].scale/2;
    fgMask(cv::Rect(offsetX, offsetY,
                    bg_w-2*offsetX, bg_h-2*offsetY))=1;

    unsigned char *pBGMask=(unsigned char *)(bgMask.data);
    unsigned char *pFGMask=(unsigned char *)(fgMask.data);
    unsigned char *pImg=(unsigned char *)(roiImg.data);
    float *pBG=(float *)(histBg.data);
    float *pFG=(float *)(histFg.data);
    int range = 256/m_trans_color_bins;
    for (int h=0; h<bg_h; h++)
    {
        for (int w=0; w<bg_w; w++)
        {
            int idx1 = pImg[w*3]/range;
            int idx2 = pImg[w*3+1]/range;
            int idx3 = pImg[w*3+2]/range;
            int idx = idx3*m_trans_color_bins*m_trans_color_bins+idx2*m_trans_color_bins+idx1;
            pBG[idx] += *(pBGMask++);
            pFG[idx] += *(pFGMask++);
        }
        pImg += roiImg.step[0];
    }
    histBg = histBg/(cv::sum(histBg)[0]);
    histFg = histFg/(cv::sum(histFg)[0]);
    if (isInit)
    {
        m_cfs[idx].bgHist = histBg.clone();
        m_cfs[idx].fgHist = histFg.clone();
    }
    else
    {
        m_cfs[idx].bgHist=(1-lr)*m_cfs[idx].bgHist+lr*histBg;
        m_cfs[idx].fgHist=(1-lr)*m_cfs[idx].fgHist+lr*histFg;
    }
    return 0;
}

int KCFExpTracker::solveTransCF(cv::Mat &num,
                             cv::Mat &den,
                             cv::Mat &fea, cv::Mat y)
{
    float norm = 1.0f/(fea.cols*fea.rows);
    std::vector<cv::Mat> feaSplit;
    cv::split(fea, feaSplit);
    
    std::vector<cv::Mat> numV;
    den = cv::Mat::zeros(fea.rows, fea.cols, CV_32F);
    for (int i=0; i<feaSplit.size(); i++)
    {
        cv::Mat feaFFT = FFTTools::fftd(feaSplit[i]);
        numV.push_back(FFTTools::complexConjMult(y,
                                                 feaFFT));
        den=den + FFTTools::complexSelfConjMult(feaFFT)*norm;
    }
    cv::merge(numV, num);
    feaSplit.clear();
    numV.clear();
    num = num*norm;
    return 0;
}

int KCFExpTracker::applyHann2D(cv::Mat& fea, cv::Mat& hann)
{
    int channels = fea.channels();
    int width = fea.cols;
    int height = fea.rows;
    float *pFea = (float *)(fea.data);
    float *pHann = (float *)(hann.data);
    for (int h=0; h<height; h++)
    {
        for (int w=0; w<width; w++)
        {
            float factor = *(pHann++);
            for (int c=0; c<channels; c++)
                pFea[c] = pFea[c] * factor;
            pFea += channels;
        }
    }
    return 0;
}

cv::Mat KCFExpTracker::createHann2D(int height,
                                 int width)
{   
    cv::Mat hann1t = cv::Mat(cv::Size(width,1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,height), CV_32F, cv::Scalar(0)); 
    
    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * PI_T * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * PI_T * i / (hann2t.rows - 1)));

    return hann2t*hann1t;
}

cv::Mat KCFExpTracker::createGaussPeak(int idx, int sizey, int sizex, float sigma)
{
    cv::Mat_<float> res(sizey, sizex);
    
    int syh = sizey/2;
    int sxh = sizex/2;

    int tw = round(m_cfs[idx].target_size[0]*m_cfs[idx].scale);
    int th = round(m_cfs[idx].target_size[1]*m_cfs[idx].scale);
    float output_sigma = std::sqrt((float)th*tw)*sigma/m_trans_cell_size;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res.at<float>(i, j) = std::exp(mult * (ih*ih+jh*jh));
        }

    // circshift the data
    //    cv::Mat resCS = res.clone();
    // for (int h=0; h<sizey; h++)
    // {
    //     int idy = (h+syh)%sizey;
    //     for (int w=0; w<sizex; w++)  
    //     {    
    //         int idx = (w+sxh)%(sizex);
    //         //resCS.at<float>(idy,idx) = res.at<float>(h,w);
    //         resCS.at<float>(h,w) = res.at<float>(idy,idx);
    //     }
    // }
    return FFTTools::fftd(res);
}

cv::Mat KCFExpTracker::cropTransResponseCF(int idx,
                                        cv::Mat &res)
{
    cv::resize(res, res,
               cv::Size(m_cfs[idx].norm_delta_size,
                        m_cfs[idx].norm_delta_size));
    return res;
}

cv::Mat KCFExpTracker::cropTransResponsePWP(int idx,
                                         cv::Mat &res)
{

    cv::Mat II = cv::Mat(cv::Size(res.cols+1,
                                  res.rows+1),
                         CV_32F);

    integral<float, float>((float *)(res.data),
                           res.cols, res.cols, res.rows,
                           (float *)(II.data));
    int tw = round(m_cfs[idx].target_size[0]*m_cfs[idx].scale);
    int th = round(m_cfs[idx].target_size[1]*m_cfs[idx].scale);
    float factor = 1.f/(tw*th);
    
    cv::Mat newRes=cv::Mat(cv::Size(m_cfs[idx].norm_delta_size,m_cfs[idx].norm_delta_size),
                           CV_32F);
    
    float *pDst = (float *)(newRes.data);
    for (int h=0; h<newRes.rows; h++)
    {
        for (int w=0; w<newRes.cols; w++)
        {
            float val = II.at<float>(h,w)+            \
                II.at<float>(h+th,w+tw)-              \
                II.at<float>(h,w+tw)   -              \
                II.at<float>(h+th,w);
            *(pDst++) = val*factor;
        }
    }
    return newRes;
}

int KCFExpTracker::detectScaleCF(int idx, float &conf)
{
    cv::Mat fea = getScaleFeaCF(idx);
    fea = FFTTools::fftd1d(fea, 1);

    cv::Mat resH = cv::Mat(cv::Size(m_scale_num, 1),
                           CV_32FC2);    
    float *ptr1, *ptr2, *ptr3, *ptr4;
    ptr1 = (float *)(resH.data);
    ptr2 = (float *)(fea.data);
    ptr3 = (float *)(m_cfs[idx].num_scale.data);
    ptr4 = (float *)(m_cfs[idx].den_scale.data);

    for (int i=0; i<m_scale_num; i++)
    {
        double realV=0, complV=0;
        for (int j=0; j<fea.cols; j++)
        {
            realV += ptr2[0]*ptr3[0] - ptr2[1]*ptr3[1];
            complV+= ptr2[0]*ptr3[1] + ptr2[1]*ptr3[0];
            ptr2 += 2;
            ptr3 += 2;
        }
        ptr1[0] = realV / (ptr4[i] + m_scale_lambda);
        ptr1[1] = complV/ (ptr4[i] + m_scale_lambda);
        ptr1 += 2;
    }

    cv::Mat res = FFTTools::real(FFTTools::fftd(resH,
                                                true));
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
    conf = (float)pv;
    
    float bestScale = m_scale_factors[pi.x];
    m_cfs[idx].scale_adapt *= bestScale;
    if (m_cfs[idx].scale_adapt < m_cfs[idx].scale_min_factor)
        m_cfs[idx].scale_adapt=m_cfs[idx].scale_min_factor;
    else if(m_cfs[idx].scale_adapt > m_cfs[idx].scale_max_factor)
        m_cfs[idx].scale_adapt= m_cfs[idx].scale_max_factor;

    float tz_w, tz_h, bg_w, bg_h, fg_w, fg_h;
    tz_w = m_cfs[idx].base_tz[0]*m_cfs[idx].scale_adapt;
    tz_h = m_cfs[idx].base_tz[1]*m_cfs[idx].scale_adapt;
    float pad = (tz_w+tz_h)/2.f;

    bg_w = tz_w+pad;
    bg_h = tz_h+pad;
    float scale = sqrt(m_trans_fixed_area/(bg_w*bg_h));
    int pre_norm_bg_w=round(m_cfs[idx].bg_size[0]*m_cfs[idx].scale);
    int pre_norm_bg_h=round(m_cfs[idx].bg_size[1]*m_cfs[idx].scale);
    bg_w = pre_norm_bg_w/scale;
    bg_h = pre_norm_bg_h/scale;
    fg_w = tz_w-pad*m_trans_inner_padding;
    fg_h = tz_h-pad*m_trans_inner_padding;

    // Apply the new value
    m_cfs[idx].target_size[0] = round(tz_w);
    m_cfs[idx].target_size[1] = round(tz_h);
    m_cfs[idx].bg_size[0] = bg_w;
    m_cfs[idx].bg_size[1] = bg_h;
    m_cfs[idx].fg_size[0] = fg_w;
    m_cfs[idx].fg_size[1] = fg_h;
    m_cfs[idx].scale = scale;
    return 0;
}

int KCFExpTracker::getOneScaleFeaCF(cv::Mat &roiImg,
                                 cv::Mat &feaHog)
{
    //Get HOG Feature
    cv::Mat roiGray;
    cv::cvtColor(roiImg, roiGray, CV_BGR2GRAY);

    feaHog = fhog(roiGray, m_scale_cell_size);
    return 0;
}

cv::Mat KCFExpTracker::getScaleFeaCF(int idx)
{
    int tw = m_cfs[idx].target_size[0];
    int th = m_cfs[idx].target_size[1];
    cv::Mat image = cv::Mat(m_img.height,m_img.width,
                            CV_8UC3, m_img.data[0]);

    cv::Mat feaTmp, feas;
    for (int i=0; i<m_scale_num; i++)
    {
        int scale_tw = MAX_T(floor(tw*m_scale_factors[i]),m_scale_cell_size);
        int scale_th = MAX_T(floor(th*m_scale_factors[i]),m_scale_cell_size);

        cv::Rect roi;
        roi.width = scale_tw;
        roi.height = scale_th;
        roi.x = round(m_cfs[idx].pos[0] - roi.width/2.f);
        roi.y = round(m_cfs[idx].pos[1] - roi.height/2.f);
        cv::Mat z=RectTools::subwindow(image, roi,
                                       cv::BORDER_REPLICATE);
        if (z.cols != m_cfs[idx].scale_norm_tz[0] || z.rows != m_cfs[idx].scale_norm_tz[1])
            cv::resize(z,z,cv::Size(m_cfs[idx].scale_norm_tz[0],
                                    m_cfs[idx].scale_norm_tz[1]));
        //resize(z,z,cv::Size(m_cfs[idx].scale_norm_tz[0],
        //                      m_cfs[idx].scale_norm_tz[1]));
        getOneScaleFeaCF(z, feaTmp);
        feaTmp = feaTmp.reshape(1,1);
        feaTmp = feaTmp * m_scale_hann[i];
        if (0==i)
            feas = feaTmp.clone();
        else
            feas.push_back(feaTmp);
    }
    return feas;
}

int KCFExpTracker::trainScaleCF(int idx, float lr, bool isInit)
{
    cv::Mat fea = getScaleFeaCF(idx);
    fea = FFTTools::fftd1d(fea, 1);
    
    cv::Mat num = cv::Mat(cv::Size(fea.cols, fea.rows),
                          CV_32FC2);
    cv::Mat den = cv::Mat(cv::Size(m_scale_num, 1),
                          CV_32F);
    
    float *ptr1, *ptr2, *ptr3;
    ptr1 = (float *)(den.data);
    ptr2 = (float *)(fea.data);
    for (int i=0; i<m_scale_num; i++)
    {
        double val = 0;
        for (int j=0; j<fea.cols; j++)
        {
            val += ptr2[0]*ptr2[0] + ptr2[1]*ptr2[1];
            ptr2 +=2;
        }
        ptr1[i] = val;
    }

    ptr1 = (float *)(num.data);
    ptr2 = (float *)(fea.data);
    ptr3 = (float *)(m_scale_y.data);
    for (int i=0; i<num.rows; i++)
    {
        for (int j=0; j<num.cols; j++)
        {
            ptr1[0] = ptr2[0]*ptr3[0] + ptr2[1]*ptr3[1];
            ptr1[1] = ptr2[0]*ptr3[1] - ptr2[1]*ptr3[0];
            ptr1 += 2;
            ptr2 += 2;
        }
        ptr3 += 2;
    }

    if (isInit)
    {
        m_cfs[idx].num_scale = num.clone();
        m_cfs[idx].den_scale = den.clone();
    }
    else
    {
        m_cfs[idx].num_scale=(1-lr)*m_cfs[idx].num_scale + lr*num;
        m_cfs[idx].den_scale=(1-lr)*m_cfs[idx].den_scale + lr*den;
    }
    return 0;
}
