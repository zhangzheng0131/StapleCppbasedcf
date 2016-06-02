#include <math.h>
#include "stapletracker.hpp"
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
        ptr += dst.cols*3;
    }
    return 0;
}
static int resize(cv::Mat &ori, cv::Mat &dst, cv::Size sz)
{
    cv::Mat aa = cv::Mat(sz, CV_8UC3);

    double scaleY = ori.rows*1.0/sz.height;
    double scaleX = ori.cols*1.0/sz.width;
    for (int h=0; h<sz.height; h++)
    {
        int y = MIN_T(round((h+1-0.5)*scaleY+0.5)-1,ori.rows-1);
        for (int w=0; w<sz.width; w++)
        {
            int x = MIN_T(round((w+1-0.5)*scaleX+0.5)-1,ori.cols-1);
            aa.at<cv::Vec3b>(h,w) = ori.at<cv::Vec3b>(y,x);
        }
    }
    
    dst = aa;
    return 0;
}

static int resize1C(cv::Mat &ori, cv::Mat &dst, cv::Size sz)
{
    cv::Mat aa = cv::Mat(sz, CV_32F);

    double scaleY = ori.rows*1.0/sz.height;
    double scaleX = ori.cols*1.0/sz.width;
    for (int h=0; h<sz.height; h++)
    {
        int y = MIN_T(round((h+1-0.5)*scaleY+0.5)-1,ori.rows-1);
        for (int w=0; w<sz.width; w++)
        {
            int x = MIN_T(round((w+1-0.5)*scaleX+0.5)-1,ori.cols-1);
            aa.at<float>(h,w) = ori.at<float>(y,x);
        }
    }
    
    dst = aa;
    return 0;
}

StapleTracker::StapleTracker(int maxSide, int minSide,
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

int StapleTracker::init()
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

int StapleTracker::initTransCF()
{
    return 0;
}

int StapleTracker::initScaleCF()
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

int StapleTracker::add(Rect_T &roi, int cate_id)
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

    float pad = (roi_s.w+roi_s.h)/2;
    int bg_w, bg_h, fg_w, fg_h;
    bg_w = round(roi_s.w+pad);
    bg_h = round(roi_s.h+pad);
    float scale = sqrt(m_trans_fixed_area/(bg_w*bg_h));
    bg_w = round((int(int(round(bg_w*scale))/m_trans_cell_size)/2*2+1)*m_trans_cell_size/scale);
    bg_h = round((int(int(round(bg_h*scale))/m_trans_cell_size)/2*2+1)*m_trans_cell_size/scale);
    fg_w = round(roi_s.w-pad*m_trans_inner_padding);
    fg_h = round(roi_s.h-pad*m_trans_inner_padding);
    
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

int StapleTracker::update()
{
    m_curObjNum = 0;
    for (int i=0 ;i<m_maxObjNum; i++)
    {
        if (-1 == m_objs[i].status)
            continue;

        float conf;
        detectTrans(i, conf);
        if (conf < 0.15f)
        {
            printf("trans conf %f\n", conf);
            m_objs[i].status = -1;
            continue;
        }
            
        detectScaleCF(i, conf);
        if (conf < 0.15f)
        {
            printf("scale conf %f\n", conf);
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
        cv::Mat roiImg;
        roiImg = getSubWin(i);
        trainTransCF(i, roiImg, m_trans_lr_cf, false);
        trainTransPWP(i, roiImg, m_trans_lr_pwp, false);
        trainScaleCF(i, m_scale_lr, false);
        m_curObjNum ++;
    }
    return m_curObjNum;
}

Rect_T StapleTracker::getShowRect(int idx)
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
int StapleTracker::detectTrans(int idx, float &conf)
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
    int center = (m_cfs[idx].norm_delta_size-1)/2;
    m_cfs[idx].pos[0] += (pi.x-center)/m_cfs[idx].scale;
    m_cfs[idx].pos[1] += (pi.y-center)/m_cfs[idx].scale;
    conf = (float)pv;
    return 0;
}

cv::Mat StapleTracker::detectTransCF(int idx,
                                     cv::Mat &roiImg)
{
    cv::Mat feaCF;
    getTransFeaCF(roiImg, feaCF);
    applyHann2D(feaCF, m_cfs[idx].hann);
    
    std::vector<cv::Mat> feaSplit;
    cv::split(feaCF, feaSplit);
    int ch = feaSplit.size();
    int stepCh = ch*2;
    cv::Mat resF = cv::Mat::zeros(feaCF.rows, feaCF.cols,
                                  CV_32FC2);
    for (int c=0; c<ch; c++)
    {
        cv::Mat feaFFT = FFTTools::fftd(feaSplit[c]);
        float *pFea = (float *)(feaFFT.data);
        float *pA = (float *)(m_cfs[idx].alpha.data)+c*2;
        float *pRes = (float *)(resF.data);
        for (int h=0; h<feaCF.rows; h++)
        {
            for (int w=0; w<feaCF.cols; w++)
            {
                pRes[0]=pRes[0]+pA[0]*pFea[0]+pA[1]*pFea[1];
                pRes[1]=pRes[1]+pA[0]*pFea[1]-pA[1]*pFea[0];
                pRes += 2;
                pFea += 2;
                pA += stepCh;
            }
        }
    }
    return FFTTools::real(FFTTools::fftd(resF, true));
}

cv::Mat StapleTracker::detectTransPWP(int idx,
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
            int idx1 = pImg[0]/range;
            int idx2 = pImg[1]/range;
            int idx3 = pImg[2]/range;
            int idx = idx3*m_trans_color_bins*m_trans_color_bins+idx2*m_trans_color_bins+idx1;
            float fg = pFG[idx];
            float bg = pBG[idx];
            if ((fg+bg)<0.000001f)
                *(pRes++) = 0;
            else
                *(pRes++) = fg/(fg+bg);
            pImg += 3;
        }
    }
    return res;
}

cv::Mat StapleTracker::getSubWinPWP(int idx)
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

cv::Mat StapleTracker::getSubWin(int idx)
{
    cv::Rect roi;
    float cx = m_cfs[idx].pos[0];
    float cy = m_cfs[idx].pos[1];    
    roi.width = m_cfs[idx].bg_size[0];
    roi.height = m_cfs[idx].bg_size[1];
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

int StapleTracker::getTransFeaCF(cv::Mat &roiImg,
                                 cv::Mat &feaHog)
{
    //Get HOG Feature
    cv::Mat roiGray;
    //bgr2gray(roiImg, roiGray);
    cv::cvtColor(roiImg, roiGray, CV_BGR2GRAY);
    feaHog = fhog(roiGray, m_trans_cell_size);
    return 0;
}

int StapleTracker::trainTransCF(int idx, cv::Mat &roiImg,
                                float lr, bool isInit)
{
    cv::Mat feaCF;
    getTransFeaCF(roiImg, feaCF);
    if (isInit)
    {
        m_cfs[idx].hann = createHann2D(feaCF.rows,
                                       feaCF.cols);
        m_cfs[idx].y=createGaussianPeak(idx, feaCF.rows,
                                        feaCF.cols,
                                        m_trans_y_sigma);
    }
    
    cv::Mat num, den;
    applyHann2D(feaCF, m_cfs[idx].hann);
    solveTransCF(num, den, feaCF, m_cfs[idx].y);
    
    if (isInit)
    {
        m_cfs[idx].num = num.clone();
        m_cfs[idx].den = den.clone();
        m_cfs[idx].alpha = num.clone();
    }
    else
    {
        m_cfs[idx].num = (1-lr)*m_cfs[idx].num + lr*num;
        m_cfs[idx].den = (1-lr)*m_cfs[idx].den + lr*den;
    }

    //Compute the alpha
    float *pA = (float *)(m_cfs[idx].alpha.data);
    float *pNum = (float *)(m_cfs[idx].num.data);
    float *pDen = (float *)(m_cfs[idx].den.data);
    int channels =  feaCF.channels();
    for (int h=0; h<feaCF.rows; h++)
    {
        for (int w=0; w<feaCF.cols; w++)
        {
            float factor = 1.0f/(*(pDen++)+m_trans_lambda);
            for (int c=0; c<channels; c++)
            {
                pA[0]=pNum[0]*factor;
                pA[1]=pNum[1]*factor;
                pA += 2;
                pNum += 2;
            }
        }
    }
    return 0;
}

int StapleTracker::trainTransPWP(int idx,
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
                    bg_w-2*offsetX, bg_h-2*offsetY))=0;
    
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
            int idx1 = pImg[0]/range;
            int idx2 = pImg[1]/range;
            int idx3 = pImg[2]/range;
            int idx = idx3*m_trans_color_bins*m_trans_color_bins+idx2*m_trans_color_bins+idx1;
            pBG[idx] += *(pBGMask++);
            pFG[idx] += *(pFGMask++);
            pImg += 3;
        }
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

int StapleTracker::solveTransCF(cv::Mat &num, cv::Mat &den,
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
int StapleTracker::applyHann2D(cv::Mat& fea, cv::Mat& hann)
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

cv::Mat StapleTracker::createHann2D(int height,
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

cv::Mat StapleTracker::createGaussianPeak(int idx, int sizey, int sizex, float sigma)
{
    cv::Mat_<float> res(sizey, sizex);
    
    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

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
    cv::Mat resCS = res.clone();
    for (int h=0; h<sizey; h++)
    {
        int idy = (h+syh)%sizey;
        for (int w=0; w<sizex; w++)  
        {    
            int idx = (w+sxh)%(sizex);
            resCS.at<float>(idy,idx) = res.at<float>(h,w);
        }
    }    
    cv::Mat resF = FFTTools::fftd(resCS);
    return resF;
}

cv::Mat StapleTracker::cropTransResponseCF(int idx,
                                           cv::Mat &res)
{
    int size = 2*floor(((m_cfs[idx].norm_delta_size/m_trans_cell_size)-1)/2) + 1;
    int sizeH = int(size/2);
    cv::Mat newRes = cv::Mat(cv::Size(size, size),
                             CV_32F);
    float *pDst = (float *)(newRes.data);
    for (int h=0; h<newRes.rows; h++)
    {
        int idy = (res.rows+h-sizeH-1)%res.rows;
        for (int w=0; w<newRes.cols; w++)
        {
            int idx = (res.cols+w-sizeH-1)%res.cols;
            *(pDst++) = res.at<float>(idy, idx);
        }
    }

    if (m_trans_cell_size > 1)
        cv::resize(newRes, newRes, cv::Size(m_cfs[idx].norm_delta_size,m_cfs[idx].norm_delta_size));
        //resize1C(newRes, newRes, cv::Size(m_cfs[idx].norm_delta_size,m_cfs[idx].norm_delta_size));
    return newRes;
}

cv::Mat StapleTracker::cropTransResponsePWP(int idx,
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

int StapleTracker::detectScaleCF(int idx, float &conf)
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

    int bg_w, bg_h, fg_w, fg_h;
    float tz_w = m_cfs[idx].base_tz[0]*m_cfs[idx].scale_adapt;
    float tz_h = m_cfs[idx].base_tz[1]*m_cfs[idx].scale_adapt;
    float pad = (tz_w+tz_h)/2.f;
    bg_w = round(tz_w+pad);
    bg_h = round(tz_h+pad);
    float scale = sqrt(m_trans_fixed_area/(bg_w*bg_h));

    int pre_norm_bg_w = round(m_cfs[idx].bg_size[0]*m_cfs[idx].scale);
    int pre_norm_bg_h = round(m_cfs[idx].bg_size[1]*m_cfs[idx].scale);
    bg_w = round(pre_norm_bg_w/scale);
    bg_h = round(pre_norm_bg_h/scale);
    fg_w = round(tz_w-pad*m_trans_inner_padding);
    fg_h = round(tz_h-pad*m_trans_inner_padding);
    
    m_cfs[idx].target_size[0] = round(tz_w);
    m_cfs[idx].target_size[1] = round(tz_h);
    m_cfs[idx].bg_size[0] = bg_w;
    m_cfs[idx].bg_size[1] = bg_h;
    m_cfs[idx].fg_size[0] = fg_w;
    m_cfs[idx].fg_size[1] = fg_h;
    m_cfs[idx].scale = scale;
    return 0;
}

int StapleTracker::getOneScaleFeaCF(cv::Mat &roiImg,
                                    cv::Mat &feaHog)
{
    //Get HOG Feature
    cv::Mat roiGray;
    cv::cvtColor(roiImg, roiGray, CV_BGR2GRAY);
    //bgr2gray(roiImg, roiGray);
    // FILE *fid1 = fopen("./rgb.txt", "w");
    // FILE *fid2 = fopen("./gray.txt", "w");
    // uint8_t *data=(uint8_t *)(roiImg.data);
    // for (int h=0; h<roiImg.rows; h++)
    // {
    //     for (int w=0; w<roiImg.cols; w++)
    //         fprintf(fid1, "%d, %d, %d, ", data[w*3+2],
    //                 data[w*3+1], data[w*3]);
    //     data += roiImg.cols*3;
    //     fprintf(fid1, "\n");
    // }
    // fclose(fid1);

    // data=(uint8_t *)(roiGray.data);
    // for (int h=0; h<roiImg.rows; h++)
    // {
    //     for (int w=0; w<roiImg.cols; w++)
    //         fprintf(fid2, "%d, ", data[w]);
    //     data += roiImg.cols;
    //     fprintf(fid2, "\n");
    // }
    // fclose(fid2);
    
    feaHog = fhog(roiGray, m_scale_cell_size);
    return 0;
}

cv::Mat StapleTracker::getScaleFeaCF(int idx)
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

int StapleTracker::trainScaleCF(int idx, float lr, bool isInit)
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
