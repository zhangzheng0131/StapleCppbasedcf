#include <math.h>
#include "stapletracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "dollar_hog.hpp"
#include "labdata.hpp"
#include "comdef.h"
#include "integral.hpp"
#include "labdata.hpp"

/*  Test API for Matlab
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
*/

// According 3 points to get the peak of parabola
static float subPixelPeak(float left, float center, float right)
{
    float divisor = 2*center - right - left;

    if (divisor == 0)
        return 0;
    return 0.5 * (right-left)/divisor;
}

StapleTracker::StapleTracker(int maxSide, int minSide,
                             int maxObjNum)
    :Tracker(maxSide, minSide, maxObjNum)
{
    m_trans_cell_size=4; //HoG cell size
    m_trans_inner_padding = 0.2f;
    m_trans_fixed_area=150*150; //
    m_trans_color_bins = 16;
    m_trans_lr_pwp = 0.04f;
    m_trans_lr_cf = 0.01f;
    m_trans_lambda = 1e-3f; //regularization factor
    m_trans_merge_factor = 0.3f;
    m_trans_y_sigma = 1.f/16;

    //Scale parameter
    m_scale_num = 5;//orginal 33
    m_scale_cell_size = 4;
    m_scale_max_area = 32*16;
    m_scale_lr = 0.025;
    m_scale_lambda = 1e-3f;
    m_scale_step = 1.01; //orginal 1.02
    m_scale_y_sigma = 1.f/4;
    m_scale_hann = 0;
    m_scale_factors = 0;
    // by zz in 2017/05/15
    float BACF_lr=0.0125;


    //
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

    trainTransCF(idx, roiImg, 1.f, true,0.0125);//BACF_lr=0.0125
    trainTransPWP(idx, roiImg, 1.f, true);
    trainScaleCF(idx, 1.f, true);
    m_curObjNum ++;
    return 0;
}

int StapleTracker::update()
{
    m_curObjNum = 0;
#ifdef __BENCHMARK
    float train_th=0.0f, update_th=0.0f;
#else
    float train_th=0.2f, update_th=0.15f;
#endif
    float confT, confS;
    for (int i=0 ;i<m_maxObjNum; i++)
    {
        if (-1 == m_objs[i].status)
            continue;

        detectTrans(i, confT);
        if (confT < update_th)
        {
            printf("trans conf %f\n", confT);
            m_objs[i].status = -1;
            continue;
        }

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
        if (confT > train_th)
        {
            cv::Mat roiImg;
            roiImg = getSubWin(i);
            trainTransCF(i, roiImg, m_trans_lr_cf, false,0.0125);
            trainTransPWP(i, roiImg, m_trans_lr_pwp,false);
            if (confS > train_th)
                (i, m_scale_lr, false);
        }
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
    resCF = resPWP;
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
    int center = (m_cfs[idx].norm_delta_size-1)/2;
    m_cfs[idx].pos[0] += (pf.x-center)/m_cfs[idx].scale;
    m_cfs[idx].pos[1] += (pf.y-center)/m_cfs[idx].scale;

    // Make sure the bndbox is not gone outside 3/4
    int tw2 = m_cfs[idx].target_size[0]/4;
    int th2 = m_cfs[idx].target_size[1]/4;
    m_cfs[idx].pos[0] = MIN_T(MAX_T(m_cfs[idx].pos[0],
                                    1-tw2),
                              m_img.width+tw2-1);
    m_cfs[idx].pos[1] = MIN_T(MAX_T(m_cfs[idx].pos[1],
                                    1-th2),
                              m_img.height+th2-1);
    conf = (float)pv;
    return 0;
}

cv::Mat StapleTracker::detectTransCF(int idx,
                                     cv::Mat &roiImg)
{
    cv::Mat feaCF;
    getTransFeaCF(roiImg, feaCF);
    applyHann2D(feaCF, m_cfs[idx].hann);

    cv::Mat  denzz = cv::Mat::zeros(feaCF.rows, feaCF.cols, CV_32F);

    std::vector<cv::Mat> feaSplit;
    std::vector<cv::Mat> feaSplit1;
    cv::split(feaCF, feaSplit);
    cv::split(m_cfs[idx].df, feaSplit1);
    int ch = floor(feaSplit.size()/2);
    int stepCh = ch*2;
    cv::Mat resF = cv::Mat::zeros(feaCF.rows, feaCF.cols,
                                  CV_32FC2);
    for (int c=0; c<ch; c++)
    {
        cv::Mat feaFFT = FFTTools::fftd(feaSplit[c]);
        float *pFea = (float *)(feaFFT.data);

        //float *pA = (float *)(m_cfs[idx].alpha.data)+c*2;
        //for(int n=0;n<m_cfs[idx].df.size();n++)
        {
        float *pA = (float *)((feaSplit[c].data)+c*2);/////???????/?/????why +c*2?
        float *pRes = (float *)(resF.data);
        for (int h=0; h<feaCF.rows; h++)
        {
            for (int w=0; w<feaCF.cols; w++)
            {
                pRes[0] += pA[0]*pFea[0]+pA[1]*pFea[1];
                pRes[1] += pA[0]*pFea[1]-pA[1]*pFea[0];
                pRes += 2;
                pFea += 2;
                pA += stepCh;
            }
        }
      }
        denzz=denzz+FFTTools::real(FFTTools::fftd(resF, true));// by zhangzheng in 2017/5/24
    }



    return denzz;  // by zhangzheng in 2017/5/24
    //return FFTTools::real(FFTTools::fftd(resF, true));
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

int StapleTracker::getTransFeaCF(cv::Mat &roiImg,
                                 cv::Mat &feaHog)
{
    // Extract HOG Feature
    cv::Mat roiGray;
    cv::cvtColor(roiImg, roiGray, CV_BGR2GRAY);
    feaHog = fhog(roiGray, m_trans_cell_size);

#ifdef ENABLE_LAB_TRANS
    // Extract LAB Feature
    int cell_sizeQ = m_trans_cell_size*m_trans_cell_size;
    cv::Mat imgLab;
    cv::cvtColor(roiImg, imgLab, CV_BGR2Lab);
    unsigned char *pLAB = (unsigned char*)(imgLab.data);
    cv::Mat labFea = cv::Mat::zeros(cv::Size(feaHog.cols,
                                             feaHog.rows),
                                    CV_32FC(nLABCentroid));

    float *pFea = (float *)(labFea.data);
    for (int cY=0; cY<imgLab.rows; cY+=m_trans_cell_size)
    {
        for (int cX=0;cX<imgLab.cols;cX+=m_trans_cell_size)
        {
            for(int y=cY; y<cY+m_trans_cell_size; ++y)
            {
                for(int x=cX; x<cX+m_trans_cell_size; ++x)
                {
                    int idx = (imgLab.cols * y + x) * 3;
                    float l = (float)pLAB[idx];
                    float a = (float)pLAB[idx + 1];
                    float b = (float)pLAB[idx + 2];

                    // Iterate trough each centroid
                    float minDist = FLT_MAX;
                    int minIdx = 0;
                    for(int k=0; k<nLABCentroid; ++k)
                    {
                        float ld=(l-pLABCentroids[3*k]);
                        float ad=(a-pLABCentroids[3*k+1]);
                        float bd=(b-pLABCentroids[3*k+2]);

                        float dist =ld*ld + ad*ad + bd*bd;
                        if(dist < minDist){
                            minDist = dist;
                            minIdx = k;
                        }
                    }
                    pFea[minIdx] += 1.f/cell_sizeQ;
                }
            }
            pFea += nLABCentroid;
        }
    }
    std::vector<cv::Mat> fv0;
    std::vector<cv::Mat> fv1;
    cv::split(feaHog, fv0);
    cv::split(labFea, fv1);
    fv0.insert(fv0.end(), fv1.begin(), fv1.end());
    cv::merge(fv0, feaHog);
#endif
    return 0;
}



int StapleTracker::trainTransCF(int idx, cv::Mat &roiImg,
                                float lr, bool isInit,double BACF_lr)
{
    cv::Mat feaCF;

    getTransFeaCF(roiImg, feaCF);//hog ,whole real data
    if (isInit)
    {
        m_cfs[idx].hann = createHann2D(feaCF.rows,
                                       feaCF.cols);
        m_cfs[idx].y=createGaussianPeak(idx, feaCF.rows,
                                        feaCF.cols,
                                        m_trans_y_sigma);
    }
    cv::Mat num, den, numVZZ, ZX,tempnumV,tempZX,tempZZ;//by zz in 2017/5/18
    applyHann2D(feaCF, m_cfs[idx].hann);


    //std::vector<cv::Mat> feaFFT;
    cv::Mat feaFFT;
    solveTransCF(num, den, numVZZ, ZX, feaCF,  m_cfs[idx].y,feaFFT);







    //reshape by zz in 2017/5/19
  //  cv::Mat reshapezz=cv::Mat::zeros(feaCF[0].raws*feaCF[0].cols,feaCF.channels(),CV_8U);
  //  for(int n=0;n<feaCF.raws;n++)
  //  {
  //    for(int i=0;i<feaCF.cols;i++)
  //    {
  //        for(int j=0;j<feaCF.channels();j++)
  //        {
  //            reshapezz(i+n*feaCF.rows,j)=feaCF(n,i,j);
  //        }
  //  //  }
  //  }





    //

    if (isInit)
    {
        //by zz in 2017/5/18 init for some new values
        m_cfs[idx].ADMM_iteration=2;
        std::vector<cv::Mat> df1;
        std::vector<cv::Mat> sf1;
        std::vector<cv::Mat> Ldsf1;
        for(int n=0;n<feaFFT.channels();n++)
        {
        //float *Pdf=(float *)(m_cfs[idx].df.data);
        //float *Psf= (float *)(m_cfs[idx].sf.data);
        //float *PLdsf1=(float *)(m_cfs[idx].Ldsf.data);
        // for(int i=0;i<feaFFT.rows;i++)
        //{
        //        for(int j=0; j<feaFFT.cols;j++)
        //        {
        //            for(int n=0;n<feaFFT.channels();n++)
        //            {
        //                Pdf[0]=0.0;
        //                Pdf[1]=0.0;
        //                Psf[0]=0.0;
        //                Psf[1]=0.0;
        //                PLdsf1[0]=0.0;
        //                PLdsf1[1]=0.0;

        //                Pdf +=2;
        //                Psf +=2;
        //                PLdsf1 +=2;
        //            }
        //        }


            printf("feaFFT.channels() : %d\n",feaFFT.channels());
            df1.push_back(cv::Mat::zeros(feaFFT.rows,feaFFT.cols,CV_8U));
            sf1.push_back(cv::Mat::zeros(feaFFT.rows,feaFFT.cols,CV_8U));
            Ldsf1.push_back(cv::Mat::zeros(feaFFT.rows,feaFFT.cols,CV_8U));
         }

        cv::merge(df1,m_cfs[idx].df);
        cv::merge(sf1,m_cfs[idx].sf);
        cv::merge(Ldsf1,m_cfs[idx].Ldsf);
        //m_cfs[idx].df=cv::Mat::zeros(m_cfs[idx].bg_size[0],m_cfs[idx].bg_size[1],CV_8U);
        //m_cfs[idx].sf=cv::Mat::zeros(m_cfs[idx].bg_size[0],m_cfs[idx].bg_size[1],CV_8U);
        //m_cfs[idx].Ldsf=cv::Mat::zeros(m_cfs[idx].bg_size[0],m_cfs[idx].bg_size[1],CV_8U);
        m_cfs[idx].numVZZ=numVZZ.clone();
        m_cfs[idx].ZX=ZX.clone();
        //by zz in 2017/5/18 init for some new values

        m_cfs[idx].num = num.clone();
        m_cfs[idx].den = den.clone();
        m_cfs[idx].alpha = num.clone();



        //cv::Mat P=cv::Mat::zeros(cv::Mat::size(roiImg));
        //for(int i=0;i<feaCF.rows;i++)
        //{
        //    for(int j=0;i<feaCF.cols;i++)
        //    {
        //       P(i,j)=1;
        //    }
        //}
        //by zz in 2017/5/18

    }
    else
    {
        m_cfs[idx].num = (1-lr)*m_cfs[idx].num + lr*num;
        m_cfs[idx].den = (1-lr)*m_cfs[idx].den + lr*den;

        ////by zz in 2017/5/18
        tempZX=numVZZ.clone();
        tempZZ=ZX.clone();
        std::vector<cv::Mat> feasplit,feasplit1;
        cv::split(tempZX,feasplit);
        cv::split(tempZZ,feasplit1);
      //  float *pA = (float *)(m_cfs[idx].ZX.data);
        float *pnumvZZ = (float *)(m_cfs[idx].numVZZ.data);
        float *pDen = (float *)(m_cfs[idx].den.data);
        float *pZX = (float *)(m_cfs[idx].ZX.data);
        float *ptZX= (float *)(numVZZ.data);
        float *ptZZ= (float *)(ZX.data);


        int channels =  feaCF.channels();
      //  for(int i=0;i<feasplit.size();i++)
        {
          for (int h=0; h<feasplit[0].rows; h++)
          {
              for (int w=0; w<feasplit[0].cols; w++)
              {
                  for (int c=0; c<feasplit.size(); c++)
                  {
                      pnumvZZ[0]=pnumvZZ[0]*(1-BACF_lr)+BACF_lr*ptZZ[0];
                      pnumvZZ[1]=pnumvZZ[1]*(1-BACF_lr)+BACF_lr*ptZZ[1];
                      pZX[0]=ptZZ[0]*(1-BACF_lr)+BACF_lr*ptZX[0];
                      pZX[1]=ptZZ[1]*(1-BACF_lr)+BACF_lr*ptZX[1];
                      pnumvZZ += 2;
                      ptZZ += 2;
                      pZX +=2;
                      ptZX +=2;

                  }
              }
          }
          //m_cfs[idx].ZX[i].data=(1-BACF_lr)*feasplit1[i].data+BACF_lr*feasplit[i].data;
          //m_cfs[idx].numVZZ[i].data=(1-BACF_lr)*m_cfs[idx].numVZZ[i].data+BACF_lr*tempZZ[i].data;
        }
        m_cfs[idx].numVZZ=numVZZ;
        m_cfs[idx].ZX=ZX;
        //by zz in 2017/5/18
    }

    //Compute the alpha
    float *pA = (float *)(m_cfs[idx].alpha.data);
    float *pNum = (float *)(m_cfs[idx].num.data);
    float *pDen = (float *)(m_cfs[idx].den.data);
    int channels =  feaCF.channels();
    for(int iz1=0; iz1<feaCF.rows;iz1++)
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
    //
    //by zz in 2017/05/16 computing new value used in BACF
    //int MMx=feaCF.rows*feaCF.cols;
    //for (int h=0; h<feaCF.rows; h++)
    //{.
    //        for (int c=0; c<channels; c++)
    //        {
    //            pA[0]=pNum[0]*factor;
    //            pA[1]=pNum[1]*factor;
    //             pA += 2;
    //            pNum += 2;
    //        }
    //}
    int minItr = 1;
    int maxItr=2;
    int term = 1;
    int visfilt=0;
    //std::vector<cv::Mat> X=feaFFT;
    cv::Mat X=feaFFT;
    int  muo=1;
    std::vector<cv::Mat> dfo,sfo,Ldsfo;
    int Mx0;
    int Mx1;
    Mx0=X.rows;
    Mx1=X.cols;
    int Nf=feaCF.channels();
    int Mf[2];
    Mf[0]=m_cfs[idx].bg_size[0];
    Mf[1]=m_cfs[idx].bg_size[1];
    ECF(idx,muo,X,Nf,term,minItr,maxItr,visfilt);
   // m_cfs[idx].df=dfo;
   // m_cfs[idx].sf=sfo;
   // m_cfs[idx].Ldsf=Ldsfo;
    //m_cfs[idx].muo=muo;



    //cv::Mat feaCFtemp;
    //cv::resize(feaCF,feaCF,cv::Size(MMx,cv::Size(feaCF,3)));
    //std::vector<cv::Mat> feaSplit;
    //cv::split(feaCF, feaSplit);


    //for (int i=0; i<feaSplit.size(); i++) // red,green,blue 3 channels
    {
    //    cv::Mat feaFFT = FFTTools::fftd(feaSplit[i]);
    }

    //cv::resize(m_cfs[idx].ZX, newRes, cv::Size(m_cfs[idx].norm_delta_size,m_cfs[idx].norm_delta_size));//  cv:: resize  adjust size of image
    //int channels = fea.channels();
    //by zz in 2017/05/16 computing new value used in BACF
    //
    return 0;
}
//cv::Mat StapleTracker::reshape1fft(cv::Mat a)
///{
//  std::vector<cv::Mat> feaSplit;
//  cv::split(fea, feaSplit);

//  std::vector<cv::Mat> numV;
//  std::vector<cv::Mat> numVZZ21;
//  std::vector<cv::Mat> numVZX;
//
//den = cv::Mat::zeros(fea.rows, fea.cols, CV_32F);
//  for (int i=0; i<feaSplit.size(); i++)
//  {
//      cv::Mat feaFFT = FFTTools::fftd(feaSplit[i]);
//      numV.push_back(FFTTools::complexConjMult(feaFFT,
//                                               y));
//      numVZZ21.push_back(FFTTools::complexConjMult(feaFFT,
//                                               feaFFT));

//      den=den + FFTTools::complexSelfConjMult(feaFFT)*norm;
//  }
//  cv::merge(numV, num);
//  cv::merge(numVZZ21, numVZZ);
//  feaSplit.clear();
//  numV.clear();
//  numVZZ21.clear();
//  ZX=num;
//  num = num*norm;
//}
bool StapleTracker::ECF(int idx, int muo, cv::Mat &X,int Nf,int term,int minItr,int maxItr,int visfilt) {
    //  int MMx=Mx[0]*Mx[1];
    int MMx = X.rows * X.cols;
    int MMx0 = X.rows;
    int MMx1 = X.cols;
    int Nx = Nf;
    float lambda = m_trans_lambda;
    int i = 1;
    //mu
    //mumax
    //beta
    //update sf
    m_cfs[idx].ADMM_iteration=2;
    while (i <= m_cfs[idx].ADMM_iteration) {

        float *SFZ = (float *) (m_cfs[idx].sf.data);
        float *pdf = (float *) (m_cfs[idx].df.data);
        float *p = (float *) (m_cfs[idx].Ldsf.data);
        float *numZ = (float *) (m_cfs[idx].numVZZ.data);
        float *ZXZ = (float *) (m_cfs[idx].ZX.data);
        //float *pZX = (float *) (m_cfs[idx].ZX.data);
        //float *pnumVZZ = (float *) (m_cfs[idx].numVZZ.data);

        //  m_cfs[idx].numVZZ[n].data=numVZZ[n]+m_cfs[idx].muo*(cv::Mat::ones(numVZZ[n].rows,numVZZ[n].cols, numVZZ[n].channels()));
        //  }
        //  for(int n=0;n<X.size();n++)
        //  {
        //  m_cfs[idx].ZX[n]=ZX[n]+m_cfs[idx].muo*df[n]-Ldsf[n];

        //float *pDen = (float *)(m_cfs[idx].den.data);
        //int channels =  X.size();
        int izz = 0;
        for (int h = 0; h < X.rows; h++) {
            for (int w = 0; w < X.cols; w++) {
                for (int n = 0; n < Nf; n++) {
                    //by zz in 2017/5/20
                    printf("%d,%d,%d\n", ++izz,Nf,m_cfs[idx].muo);

                    numZ[0] = numZ[0] + m_cfs[idx].muo;
                    numZ[1] = numZ[1] + m_cfs[idx].muo;

                    //printf("%d");
                    if(p>(float *)(0x700000))
                    {
                        ZXZ[0]=0;
                        ZXZ[1]=0;
                        //SFZ[0]=0;
                        //SFZ[1]=0;
                    }
                    else {
                        ZXZ[0] = ZXZ[0] + float(m_cfs[idx].muo) * pdf[0] - p[0];
                        ZXZ[1] = ZXZ[1] + m_cfs[idx].muo * pdf[1] - p[1];
                        SFZ[0] = ZXZ[0] / (numZ[0]+0.00000000001);
                        SFZ[1] = ZXZ[1] / (numZ[1]+0.00000000001);
                    }
                    //
                    numZ += 2;
                    //pnumVZZ += 2;
                    //pZX += 2;
                    p += 2;
                    SFZ += 2;
                    pdf += 2;
                    ZXZ += 2;
                    //numZ += 2;
                }
            }
        }
        //reshape
        //update df
        //
        std::vector<cv::Mat> numzz, d, xf, xf1;

        int N = Nf;
        int prodMx = MMx;
        int M1 = std::floor(m_cfs[idx].target_size[0] * 0.3);
        int M2 = std::floor(m_cfs[idx].target_size[1] * 0.3);
        float at = (m_cfs[idx].muo + lambda / (std::sqrt(prodMx)));

        //xtemp[n]=cv::Mat::zeros(M1, M2, 2);
        float *pS = (float *) (m_cfs[idx].sf.data);
        float *pL = (float *) (m_cfs[idx].Ldsf.data);
        float *pT = (float *) (m_cfs[idx].df.data);

        //    temp[n]= (m_cfs[idx].muo*m_cfs[idx].sf[n].data + m_cfs[idx].Ldsf[n].data)/at;
        //  }
        for (int h = 0; h < X.rows; h++) {
            for (int w = 0; w < X.cols; w++) {
                for (int n = 0; n < Nf; n++)
                {
                    //by zz in 2017/5/20
                    if (p < (float *) (0x7bb000)) {
                   // pT[0] = (pS[0] * m_cfs[idx].muo + pL[0]) / at;
                   // pT[1] = (pS[1] * m_cfs[idx].muo + pL[1]) / at;
                    }
                    else
                    {
                        pT[0]=0;
                        pT[1]=0;
                    }
                    //  SFZ[0]=ZXZ[0]/numZ[0];
                    //  SFZ[1]=ZXZ[1]/numZ[1];
                    pT += 2;
                    pS += 2;
                    pL += 2;

                }
            }
        }
        //  numzz=xtemp;
        //cv::Mat res = FFTTools::real(FFTTools::fftd(resH,
        //                                            true));

        //for(int n=0;n<X.size();n++)
        //  {
        // cv::Mat ttemp=temp[n];
        std::vector<cv::Mat> xfzz;
        std::vector<cv::Mat>  sfzz;
        std::vector<cv::Mat> dfozz;
        std::vector<cv::Mat> split1;
        //cv::Mat temp = m_cfs[idx].df;
        //std::vector<cv::Mat> res;
        //float *PdF = (float *) (m_cfs[idx].df.data);
        //std::vector<cv::Mat> reszz;
         //   cv::Mat resH1;
         //   cv::Mat resH = cv::Mat(cv::Size(m_cfs[idx].df.rows,m_cfs[idx].df.cols),CV_32FC2);
         //   float *Pres= (float *)(resH1.data );
         //   for(int i=0;i<m_cfs[idx].df.rows;i++)
         // {
          //  for(int j=0;j<m_cfs[idx].df.cols;j++){
         //       for(int n=0;n<Nf;n++)
        //      {
         //           Pres[0]=PdF[0];
         //           Pres[1]=PdF[1];
          //          Pres+=2;
          //          PdF+=2;
          //      }
          //  }
            //reszz.push_back(FFTTools::fftd(resH, true));
        //}

        std::vector<cv::Mat> feaSplit1;
        cv::split(m_cfs[idx].df, feaSplit1);
        int ch = Nf;
        int stepCh = ch*2;
        std::vector<cv::Mat> xtemp1;
        cv::Mat xtemp;
        for (int c=0; c<ch; c++)
        {
            cv::Mat resF= cv::Mat::zeros(m_cfs[idx].df.rows, m_cfs[idx].df.cols,
                                    CV_32FC2);
            //cv::Mat feaFFT = FFTTools::fftd(feaSplit[c]);
            //float *pFea = (float *)(feaFFT.data);

            //float *pA = (float *)(m_cfs[idx].alpha.data)+c*2;
            //for(int n=0;n<m_cfs[idx].df.size();n++)
            {
                float *pA = (float *)((m_cfs[idx].df.data)+c*2);
                float *pRes = (float *)(resF.data);
                for (int h=0; h<m_cfs[idx].df.rows; h++)
                {
                    for (int w=0; w<m_cfs[idx].df.cols; w++) {
                        if (pA < (float *)(0x7a0000)) {
                        pRes[0] += pA[0];
                        pRes[1] += pA[1];
                        }
                        else
                        {
                            pRes[0]=0;
                            pRes[1]=0;
                        }
                        pRes += 2;
                        pA += stepCh;

                    }
                }
            }
            xtemp1.push_back(FFTTools::fftd(resF, true));
        }

        //cv::Mat xtemp = (FFTTools::fftd(resF, true));
        cv::merge(xtemp1,xtemp);
        xtemp1.clear();
        //std::vector<cv::Mat> split1zz;
        //cv::split(resH1,split1zz);
        //float *Pres1=(float *)(resH.data);

        //cv::Mat xtemp;
        //cv::merge(reszz,xtemp);
       // xtemp = (FFTTools::fftd(m_cfs[idx].df, true));
        //cv::split(xtemp, split1);

        //
       // cv::Mat xtemp
        int delta[2];
        delta[0] = 0 - ((std::floor(M1 - MMx0) / 2) + 1) - 1;
        delta[1] = 0 - ((std::floor(M2 - MMx1) / 2) + 1) - 1;
        cv::Mat xf2,atemp;
        xf2 = cv::Mat::zeros(MMx0, MMx1, CV_8UC2);
        for (int n = 0; n < Nf; n++) {
            //cv::Mat r=numzz(cv::Rect(delta[0],delta[1],m_cfs[idx].fg_size[0],m_cfs[idx].fg_size[1]));
            xfzz.push_back(xf2);
        }
        cv::merge(xfzz,atemp);
        xfzz.clear();
        //for (int ter = 0; ter < split1.size(); ter++){
//        RectTools::subwindow(image, roi,
//                             cv::BORDER_REPLICATE);
        cv::Mat sfo2 = RectTools::subwindow(xtemp,cv::Rect(delta[0], delta[1], M1, M2));
        //sfzz.push_back(sfo2);
        atemp(cv::Rect(delta[0], delta[1], M1, M2)) = sfo2;
        //sfzz=atemp;

        for (int c=0; c<ch; c++)
        {
            cv::Mat resF1= cv::Mat::zeros(m_cfs[idx].df.rows, m_cfs[idx].df.cols,
                                         CV_32FC2);
            {
                float *pA = (float *)((atemp.data)+c*2);
                float *pRes = (float *)(resF1.data);
                for (int h=0; h<m_cfs[idx].df.rows; h++)
                {
                    for (int w=0; w<m_cfs[idx].df.cols; w++) {
                        if (pA < (float *) (0xab0000) && pA > (float *)(0x800051)) {
                            pRes[0] += pA[0];
                            pRes[1] += pA[1];
                        }
                        else{
                            pRes[0]=0;
                            pRes[1]=0;
                        }
                            pRes += 2;
                            //pFea += 2;
                            pA += stepCh;
                    }
                }
            }
            sfzz.push_back(FFTTools::fftd(resF1, true));
        }
        //cv::Mat dfo2 = (FFTTools::fftd(atemp, true));
       // dfozz.push_back(dfo2);
      //}
        cv::merge(sfzz,m_cfs[idx].df);
        sfzz.clear();

        std::vector<cv::Mat> split4;
        cv::split(m_cfs[idx].df, split4);
        pS = (float *) (m_cfs[idx].sf.data);
        //float *pL = (float *)(m_cfs[idx].Ldsf.data);
            float *pD = (float *) (m_cfs[idx].df.data);
            //    temp[n]= (m_cfs[idx].muo*m_cfs[idx].sf[n].data + m_cfs[idx].Ldsf[n].data)/at;
            //  }
            for (int h = 0; h < X.rows; h++) {
                for (int w = 0; w <X.cols; w++) {
                    for (int n = 0; n < Nf; n++) {
                    //by zz in 2017/5/20
                    pS[0] = pD[0];
                    pS[1] = pD[1];

                    //  SFZ[0]=ZXZ[0]/numZ[0];
                    //  SFZ[1]=ZXZ[1]/numZ[1];
                    pT += 2;
                    pD += 2;
                    // pL += 2;
                }
            }
        }
        split4.clear();
            //cv::merge(sfzz,m_cfs[idx].sf);//
            // update Ldsf
            pS = (float *) (m_cfs[idx].sf.data);
            pL = (float *) (m_cfs[idx].Ldsf.data);
            pD = (float *) (m_cfs[idx].df.data);


            //    temp[n]= (m_cfs[idx].muo*m_cfs[idx].sf[n].data + m_cfs[idx].Ldsf[n].data)/at;
            //  }
            for (int h = 0; h < X.rows; h++) {
                for (int w = 0; w < X.cols; w++) {
                    for (int n = 0; n < Nf; n++) {

//Ldsfo[n]= Ldsf[n] + (m_cfs[idx].muo*(sfo[n]-dfo[n]));
                        //by zz in 2017/5/20
                        pL[0] = pL[0] + (pS[0] - pD[0]) * m_cfs[idx].muo;
                        pL[1] = pL[1] + (pS[1] - pD[1]) * m_cfs[idx].muo;


                        //  SFZ[0]=ZXZ[0]/numZ[0];
                        //  SFZ[1]=ZXZ[1]/numZ[1];
                        pD += 2;
                        pS += 2;
                        pL += 2;
                    }
                }
            }
            //cv::merge(xtemp, numzz);

            //update muo
            //cv::Mat dfo = FFTTools::fftd(xtemp);
            //Ldsfo= Ldsf + (m_cfs[idx].muo*(sfo-dfo));
            if (m_cfs[idx].muo * m_cfs[idx].beta > m_cfs[idx].mumax) {
                m_cfs[idx].muo = m_cfs[idx].mumax;
            } else {
                m_cfs[idx].muo = m_cfs[idx].muo * m_cfs[idx].beta;
            }
            //float *DFZ = (float *)(dfo.data);
    i = i + 1;
    }
    return true;
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

int StapleTracker::solveTransCF(cv::Mat &num, cv::Mat &den, cv::Mat &numVZZ, cv::Mat &ZX,
                                cv::Mat &fea, cv::Mat y, cv::Mat &feaFFT)
{
    float norm = 1.0f/(fea.cols*fea.rows);
    std::vector<cv::Mat> feaSplit;
    cv::split(fea, feaSplit);
    cv::Mat temp;

    std::vector<cv::Mat> numV;
    std::vector<cv::Mat> numVZZ21;
    std::vector<cv::Mat> numVZX;
    std::vector<cv::Mat> feaFFT1;

    den = cv::Mat::zeros(fea.rows, fea.cols, CV_32F);
    for (int i=0; i<feaSplit.size(); i++)
    {
        cv::Mat feaFFT2 = FFTTools::fftd(feaSplit[i]);
        feaFFT1.push_back(FFTTools::fftd(feaSplit[i]));

        numV.push_back(FFTTools::complexConjMult(feaFFT2,
                                                 y));
        numVZZ21.push_back(FFTTools::complexConjMult(feaFFT2,
                                                     feaFFT2));

        den=den + FFTTools::complexSelfConjMult(feaFFT2)*norm;
    }
    cv::merge(numV, num);
    cv::merge(numVZZ21, numVZZ);
    cv::merge(feaFFT1,feaFFT);
    feaSplit.clear();
    numV.clear();
    numVZZ21.clear();
    feaFFT1.clear();
    ZX=num;
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

int StapleTracker::getOneScaleFeaCF(cv::Mat &roiImg,
                                    cv::Mat &feaHog)
{
    //Get HOG Feature
    cv::Mat roiGray;
    cv::cvtColor(roiImg, roiGray, CV_BGR2GRAY);
    feaHog = fhog(roiGray, m_scale_cell_size);

#ifdef ENABLE_LAB_SCALE
    // Extract LAB Feature
    int cell_sizeQ = m_scale_cell_size*m_scale_cell_size;
    cv::Mat imgLab;
    cv::cvtColor(roiImg, imgLab, CV_BGR2Lab);
    unsigned char *pLAB = (unsigned char*)(imgLab.data);
    cv::Mat feaLab = cv::Mat::zeros(cv::Size(feaHog.cols,
                                             feaHog.rows),
                                    CV_32FC(nLABCentroid));

    float *pFea = (float *)(feaLab.data);
    for (int cY=0; cY<imgLab.rows-m_scale_cell_size; cY+=m_scale_cell_size)
    {
        for (int cX=0;cX<imgLab.cols-m_scale_cell_size;cX+=m_scale_cell_size)
        {
            for(int y=cY; y<cY+m_scale_cell_size; ++y)
            {
                for(int x=cX; x<cX+m_scale_cell_size; ++x)
                {
                    int idx = (imgLab.cols * y + x) * 3;
                    float l = (float)pLAB[idx];
                    float a = (float)pLAB[idx + 1];
                    float b = (float)pLAB[idx + 2];

                    // Iterate trough each centroid
                    float minDist = FLT_MAX;
                    int minIdx = 0;
                    for(int k=0; k<nLABCentroid; ++k)
                    {
                        float ld=(l-pLABCentroids[3*k]);
                        float ad=(a-pLABCentroids[3*k+1]);
                        float bd=(b-pLABCentroids[3*k+2]);

                        float dist =ld*ld + ad*ad + bd*bd;
                        if(dist < minDist){
                            minDist = dist;
                            minIdx = k;
                        }
                    }
                    pFea[minIdx] += 1.f/cell_sizeQ;
                }
            }
            pFea += nLABCentroid;
        }
    }
    std::vector<cv::Mat> fv0;
    std::vector<cv::Mat> fv1;
    cv::split(feaHog, fv0);
    cv::split(feaLab, fv1);
    fv0.insert(fv0.end(), fv1.begin(), fv1.end());
    cv::merge(fv0, feaHog);
#endif
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
        cv::Mat z;
        z = RectTools::subwindow(image, roi,
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
