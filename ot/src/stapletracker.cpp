#include <math.h>
#include <slapi-plugin.h>
#include "stapletracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "dollar_hog.hpp"
#include "labdata.hpp"
#include "comdef.h"
#include "integral.hpp"
#include "labdata.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "shift.hpp"

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

double pinterpScaleFactors[33];
//int frameW;
//nt frameH;
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
    m_scale_num = 9;//orginal 33 ,m_scale_num = 5;
    m_scale_cell_size = 4;
    m_scale_max_area = 32*16;
    m_scale_lr = 0.025;
    m_scale_lambda = 1e-3f;
    m_scale_step = 1.02; //orginal 1.02 1.01
    m_scale_y_sigma = 1.f/4;
    m_scale_hann = 0;
    m_scale_factors = 0;
    // by zz in 2017/05/15
    BACF_lr=0.0125;
     pfcell_size = 4;
    //m_cfs[idx]muo=1;
    //pfcell_size ;
    //mumax=1000;
    //beta=10;
    MaxItr=2;
    frameW=480;
    frameH=640;

    //
    //by zhangzheng in 2017/5/22
    //pvisualzation=1;
     visualization=0;
    debug=0;
    search_area_scale=4;
    filter_size=1.2000;
    output_sigma_factor=0.0625;
    ini_imgs=8;
    etha=0.0125;// 0.0125;
    upResize=100;
    lowResize=50;
    term=0.000001;
    lambda=0.001;
    slambda=0.0100;
    //beta=10;
    mu=1;
    maxMu=1000;
    search_area_shape="sequre";
    search_area_scale=4;
    min_image_sample_size=40000;
    max_image_sample_size=90000;
    fix_model_size=6400;
     pfcolorUpdataRate=0.0100;
     pft[0]="greyHoG";
     pft[1]="grey";
     pfhog_orientation=9;
     pfnbin=10;
     cell_size=4;
    //cv::Mat pfw2c==;
    //struct pfcolorTransform{


    //};
     pfinterPatchRate=0.3;
     pfgrey=1;
     pfgreyHoG=1;
     pfcolorProb=0;
     pfcolorProbHoG=0;
     pfcolorName=0;
     pfgreyProb=0;
     pflbp=0;
     nScales=17;
     nScalesInterp=33;
     scale_step=1.02;
     scale_sigma_factor=0.0625;
     scale_model_factor=1;
    scale_model_max_area=512;
     s_num_compressed_dim="MAX";
     interp_factor=0.0250;
     search_size[0] =1;
     search_size[1]=0.9850;
     search_size[2]=0.9900;
     search_size[3]=0.9950;
     search_size[4]=1.0050;
     search_size[5]=1.0100;
     search_size[6]=1.0150;
     resize_image=0;
     resize_scale=1;
   // pscaleSizeFactors=cv::Seros();
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
    //if (0==m_scale_hann)
   // {
   //     m_scale_hann = new float[m_scale_num];
   //     if (0==m_scale_hann)
   //         return -1;
   //     for (int i = 0; i < m_scale_num; i++)
   //         m_scale_hann[i] = 0.5*(1-std::cos(2*PI_T*i/(m_scale_num-1)));
   // }

   // if (0==m_scale_factors)
   // {
   //     m_scale_factors = new float[m_scale_num];
    //    for (int i = 0; i < m_scale_num; i++) {
           // m_scale_factors[i] = pow(m_scale_step,
           //                          int(m_scale_num / 2) - i);
    //        m_scale_factors[i] = pow(m_scale_step,
    //                                (i - int(m_scale_num / 2))*nScalesInterp/nScales );
    //    }
    //}

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

int StapleTracker::add(Rect_T &roi, int cate_id) {
    Rect_T roi_s, winR;
    float model_area=roi.w*roi.h*filter_size*filter_size;
    float model_currentScaleFactor= sqrt(model_area/fix_model_size);
    roi_s.x = (int) (roi.x );
    roi_s.y = (int) (roi.y );
    //roi_s.w = ((int) (roi.w / m_scale)) / 2 * 2;
    roi_s.w= (int) (roi.w);
    roi_s.h =(int) (roi.h);

    //roi_s.h = ((int) (roi.h / m_scale)) / 2 * 2;

    int idx = getIdleIdx();
    if (-1 == idx)
        return 0;

    if (1 == isAlreadyIn(roi_s))
        return 0;

    float pad = (roi_s.w + roi_s.h) / 2.f;
    float bg_w, bg_h, fg_w, fg_h;
    bg_w = round(roi_s.w + pad);
    bg_h = round(roi_s.h + pad);
    float scale = sqrt(m_trans_fixed_area / (bg_w * bg_h));
    bg_w = (int(int(round(bg_w * scale)) / m_trans_cell_size) / 2 * 2 + 1) * m_trans_cell_size / scale;
    bg_h = (int(int(round(bg_h * scale)) / m_trans_cell_size) / 2 * 2 + 1) * m_trans_cell_size / scale;
    fg_w = roi_s.w - pad * m_trans_inner_padding;
    fg_h = roi_s.h - pad * m_trans_inner_padding;

    m_cfs[idx].pos[0] = roi_s.x + roi_s.w / 2;
    m_cfs[idx].pos[1] = roi_s.y + roi_s.h / 2;
    m_cfs[idx].target_size[0] = roi_s.w;
    m_cfs[idx].target_size[1] = roi_s.h;
    m_cfs[idx].base_tz[0] = roi_s.w;
    m_cfs[idx].base_tz[1] = roi_s.h;
    m_cfs[idx].bg_size[0] = bg_w;
    m_cfs[idx].bg_size[1] = bg_h;
    m_cfs[idx].fg_size[0] = fg_w;
    m_cfs[idx].fg_size[1] = fg_h;
    m_cfs[idx].scale = scale;
    frameW;
   // m_cfs[idx].rate2img[0]=m_img.width * 1.0f/ frameW;
  //  m_cfs[idx].rate2img[1]=m_img.height *1.0f/ frameH;
    float radius = MIN_T((bg_w - roi_s.w) * m_cfs[idx].scale,
                         (bg_h - roi_s.h) * m_cfs[idx].scale);
    m_cfs[idx].norm_delta_size = 2 * floor(radius / 2) + 1;

    m_objs[idx].roi = roi_s;
    m_objs[idx].cate_id = cate_id;
    m_objs[idx].obj_id = m_accumObjId++;
    m_objs[idx].status = 1;


    //Set the scale parameters
    float factor = 1;
    int norm_tw = round(roi_s.w * m_cfs[idx].scale);
    int norm_th = round(roi_s.h * m_cfs[idx].scale);
    if (norm_tw * norm_th > m_scale_max_area)
        factor = sqrt(m_scale_max_area / (norm_tw * norm_th));
    m_cfs[idx].scale_norm_tz[0] = floor(norm_tw * factor);
    m_cfs[idx].scale_norm_tz[1] = floor(norm_th * factor);
    m_cfs[idx].scale_max_factor = pow(m_scale_step,
                                      floor(log(MIN_T(m_img.width * 1.f / roi_s.w, m_img.height * 1.f / roi_s.h)) /
                                            log(m_scale_step)));
    m_cfs[idx].scale_min_factor = pow(m_scale_step,
                                      ceil(log(MAX_T(5.f / bg_w, 5.f / bg_h)) / log(m_scale_step)));
    m_cfs[idx].scale_adapt = 1;
    //Train the trackers

    float t = sqrt(m_cfs[idx].target_size[0] * m_cfs[idx].target_size[1]);
    bool  resize_image1;
    int resize_scale1;
    if (t >= lowResize && t < upResize)
    {
        resize_image1 = 0;
        resize_scale1=1;
    } else if(sqrt(m_cfs[idx].target_size[0] * m_cfs[idx].target_size[1])>= upResize){
        resize_image1;
        resize_image1 = 0;
        resize_scale1 = 1;
    }
    else
    {
        resize_image1 =0;
        resize_scale1 = 1 ;
    }
    resize_image=resize_image1;
    resize_scale=resize_scale1;

    if(resize_image)
    {
        //cv::Mat img=cv::resize(m_img,1/resize_scale);
    }

    float search_area= m_cfs[idx].target_size[0]*m_cfs[idx].target_size[1]*filter_size*filter_size;

    float currentScaleFactor = sqrt(search_area/fix_model_size);
    //cv::Mat mfirstImg = m_img;
    float target_sz[2];
    target_sz[0]=m_cfs[idx].target_size[0]/currentScaleFactor;
    target_sz[1]=m_cfs[idx].target_size[1]/currentScaleFactor;

    //'square'
    float sz1[2];
    sz1[0]=sqrt(target_sz[0]*target_sz[1])*search_area_scale;
    sz1[1]=sqrt(target_sz[0]*target_sz[1])*search_area_scale;
    int sz[2];
    sz[0]=round(sz1[0]);
    sz[1]=round(sz1[1]);
    int tsz= round(target_sz[0]);
    sz[0]=sz[0] - (sz[0]-tsz)%2;
    tsz = round(target_sz[1]);
    sz[1]=sz[1] - (sz[1]-tsz)%2;


    //int s_filt_sz[2] ;
    m_cfs[idx].s_filt_sz[0] = floor(target_sz[0]*filter_size);
    m_cfs[idx].s_filt_sz[1] = floor(target_sz[1]*filter_size);
    m_cfs[idx].s_filt_sz[0] = floor(m_cfs[idx].s_filt_sz[0]/pfcell_size);
    m_cfs[idx].s_filt_sz[1] = floor(m_cfs[idx].s_filt_sz[1]/pfcell_size);

    int b_filt_sz[2];
    b_filt_sz[0] = floor(sz[0]/pfcell_size);
    b_filt_sz[1] = floor(sz[1]/pfcell_size);
    m_cfs[idx].b_filt_sz[0]=b_filt_sz[0];
    m_cfs[idx].b_filt_sz[1]=b_filt_sz[1];

    float output_sigma= sqrt(m_cfs[idx].s_filt_sz[0]*m_cfs[idx].s_filt_sz[1])*output_sigma_factor;
    //m_cfs[idx].scale = currentScaleFactor;
    m_cfs[idx].scale=1;
            int patch_sz[2];
    patch_sz[0]=floor(sz[0]*currentScaleFactor);
    patch_sz[1]=floor(sz[1]*currentScaleFactor);

    if(patch_sz[0]<1) patch_sz[0]=2;
    if(patch_sz[1]<1) patch_sz[1]=2;
    m_cfs[idx].currentScaleFactor=currentScaleFactor;
    //m_cfs[idx].bg_size[0]=patch_sz[0];
    //m_cfs[idx].bg_size[1]=patch_sz[1];
    //m_cfs[idx].bg_size[0]=bg_w;
    //m_cfs[idx].bg_size[ 1]=bg_h;
    m_cfs[idx].window_sz[0]=sz[0];
    m_cfs[idx].window_sz[1]=sz[1];
    m_cfs[idx].target_size1[0]=(float)(target_sz[0]);
    m_cfs[idx].target_size1[1]=(float)(target_sz[1]);
   // m_cfs[idx].last_target_sz[0]=(float)(target_sz[0]);
    //m_cfs[idx].last_target_sz[1]=(float)(target_sz[1]);
   // m_cfs[idx].last_pos[0]=m_cfs[idx].pos[0] ;
   // m_cfs[idx].last_pos[1]=m_cfs[idx].pos[1] ;
   // m_cfs[idx].s_filt_sz[0]=;
   // pbase_target_sz[0]=(float)(target_sz[0]);
   // pase_target_sz[1]=(float)(target_sz[1]);
   // pfeatures_sz[0]=b_filt_sz[0];
  //  pfeatures_sz[1]=b_filt_sz[1];
  //  psz[0]=sz[0];
  //  psz[1]=sz[1];

    float scale_sigma = nScalesInterp * scale_sigma_factor;

//float temp= nScalesInterp/nScales;
    float scale_exp[nScales];
    for (int i = 0; i < nScales; i++) {
        scale_exp[i] = (-floor((nScales - 1) / 2) + i) * nScalesInterp / nScales;
    }
    float scale_exp_shift[nScales];
   // shift(scale_exp, scale_exp_shift,[0 - floor((nScales - 1) / 2)]);


    int interp_scale_exp[nScalesInterp];
    for (int i = 0; i < nScalesInterp; i++)
    {
        interp_scale_exp[i]=-floor((nScalesInterp-1)/2)+i;
    }
    int interp_scale_exp_shift1[nScalesInterp];
  //  shift(interp_scale_exp,interp_scale_exp_shift,[0 -floor((nScalesInterp-1)/2)]);
   // printf("\n");
    for(int i=0;i<nScalesInterp;i++)
    {
        int j=floor(nScalesInterp-(int)((i+0.5*nScalesInterp))%nScalesInterp);
        if(i<nScales)
        {
            interp_scale_exp_shift1[i]=i;
        }
        else{
            interp_scale_exp_shift1[i]=i-nScalesInterp;
        }
        //interp_scale_exp_shift1[i]=interp_scale_exp[j];
       // printf("%d\t",interp_scale_exp_shift1[i]);
    }
   // printf("\n");
    for(int i=0;i<nScales;i++)
    {
        pscaleSizeFactors[i]=pow(scale_step,scale_exp[i]);
    }
    for(int i=0;i<nScalesInterp;i++)
    {
        pinterpScaleFactors[i]=std::pow((double)(1.0f*scale_step),(double)(1.0f*interp_scale_exp_shift1[i]));
      //  printf("%.8lf\n",pinterpScaleFactors[i]);
      //  printf("%.8lf\t", std::pow((double)(1.0f*scale_step),(double)(1.0f*interp_scale_exp_shift1[i])));
    }
   // if(pow(scale_model_factor,2)*m_cfs[idx].target_size[0]*m_cfs[idx].target_size[1]>scale_model_max_area)
   // {
   //     scale_model_factor= sqrt(scale_model_max_area/(m_cfs[idx].target_size[0]*m_cfs[idx].target_size[1]>));
   // }
   // pscale_model_sz[0]=floor(m_cfs[idx].target_size[0]*scale_model_factor);
   // pscale_model_sz[1]=floor(m_cfs[idx].target_size[1]*scale_model_factor);
   // pmin_scale_factor=pow(scale_step,ceil(log(max(5/psz[0], 5/psz[1])))/log(scale_step));
  //  pmax_scale_factor=pow(scale_step,floor(log(min(roi.h/target_sz[0],roi.w/target_sz[1]))));
  //  ps_num_compressed_dim=nScales;
    //m_cfs[idx].
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
           // trainScaleCF(i,1.0f,false);
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
    //int frame=imread(roiImg);
   // cvNamedWindow("tracker roiImg",1);
    //cv::imshow("tracker roiImg",roiImg);
   // cv::imshow( "Tracker", frame);
    //cv::waitKey(10);

    cv::Mat resCF1 = detectTransCF(idx, roiImg);
    resCF1 = cropTransResponseCF(idx, resCF1);


    // Detect by translation PWP
    roiImg = getSubWinPWP(idx);
    cv::Mat resPWP = detectTransPWP(idx, roiImg);
    resPWP = cropTransResponsePWP(idx, resPWP);
    cv::resize(resPWP,resPWP,cv::Size(resCF1.rows,resCF1.cols));
    //minMax
    double maxcf,maxpwp;
    cv::minMaxLoc(resCF1,NULL,&maxcf,NULL,NULL);
    cv::minMaxLoc(resPWP,NULL,&maxpwp,NULL,NULL);
    //cv::Mat resCF = 0.7f*maxpwp/maxcf*resCF1;//+0.3f*resPWP;
    cv::Mat resCF = resCF1;
    float *PCF= (float *)(resCF1.data);
    float *PPWP = (float *)(resPWP.data);
    float *Pres = (float *)(resCF.data);
   // for(int i=0;i<resCF.rows;i++)
   // {
   //     for(int n=0;n< resCF.cols;n++)
   //     {
   //         printf("resCF1: %f \t",PCF[0]);
   //         PCF +=1;
   //     }
   //     printf("\n");
   // }
    /*for(int i=0;i<resCF.rows;i++)
    {
        for(int n=0;n< resCF.cols;n++)
        {
            printf("resCPWP: %f \t",PPWP[0]);
            PPWP +=1;
        }
        //printf("\n");
    }
    for(int i=0;i<resCF.rows;i++)
    {

        for(int n=0;n< resCF.cols;n++)
        {
            printf("resCF1: %f \t",Pres[0]);
            Pres +=1;
        }
        printf("\n");
    }*/
   // resCF = resPWP;
    cv::Point2i pi,pipwp;
    double pv,pvpwp;
    cv::minMaxLoc(resCF, NULL, &pv, NULL, &pi);
    cv::minMaxLoc(resPWP, NULL, &pvpwp, NULL, &pipwp);
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
    int center= floor((resCF.rows+resCF.cols)/4);
    //int center = 0;
    pf.x=pi.x;
    pf.y = pi.y;

    //if(pf.x>= ceil(resCF.cols/2)){pf.x= pf.x-resCF.cols;}
    //if(pf.y>= ceil(resCF.rows/2)){pf.y= pf.y - resCF.rows;}
   // m_cfs[idx].pos[0] += (pf.x-1-center)*m_cfs[idx].currentScaleFactor*m_scale_cell_size;
   // m_cfs[idx].pos[1] += (pf.y-1-center)*m_cfs[idx].currentScaleFactor*m_scale_cell_size;



    m_cfs[idx].pos[0] += (pf.x-1-center)/m_cfs[idx].scale*m_cfs[idx].rate2img[0];
    m_cfs[idx].pos[1] += (pf.y-1-center)/m_cfs[idx].scale*m_cfs[idx].rate2img[1];

   // float pos0=m_cfs[idx].pos[0]/m_cfs[idx].rate2img[0];
  //  float pos1=m_cfs[idx].pos[1]/m_cfs[idx].rate2img[1];
  //  float width0=tw/m_cfs[idx].rate2img[0];
  //  float height0=th/m_cfs[idx].rate2img[1];

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
    int ch = feaSplit.size();
    int stepCh = ch*2;

    std::vector<cv::Mat> splidf;
    cv::split(m_cfs[idx].df,splidf);
    cv::Mat resF = cv::Mat::zeros(feaCF.rows, feaCF.cols,
                                  CV_32FC2);
    for (int c=0; c<ch; c++) {
        cv::Mat feaFFT = FFTTools::fftd(feaSplit[c]);
        float *pFea = (float *) (feaFFT.data);

        //float *pA = (float *)(m_cfs[idx].alpha.data)+c*2;
        //for(int n=0;n<m_cfs[idx].df.size();n++)
        {
            float *pA = (float *) (m_cfs[idx].df.data) + c * 2;
            float *pRes = (float *) (resF.data);
            for (int h = 0; h < feaCF.rows; h++) {
                for (int w = 0; w < feaCF.cols; w++) {
                    //pRes[0] += pA[0] * pFea[0] - pA[1] * pFea[1];

                    //pRes[1] += pA[0] * pFea[1] + pA[1] * pFea[0];

                    pRes[0] += splidf[2*c].at<float>(h,w) * pFea[0] - splidf[2*c+1].at<float>(h,w) * pFea[1];

                    pRes[1] += splidf[2*c].at<float>(h,w) * pFea[1] + splidf[2*c+1].at<float>(h,w) * pFea[0];
                    //printf("res:%f+%f i|ertrttdr df: %f + %fi  \\ fea: %f+ %f i\n", pRes[0], pRes[1], pA[0], pA[1],
                     //      pFea[0], pFea[1]);
                    pRes += 2;
                    pFea += 2;
                    pA += stepCh;
                }
            }
        }
    }


            /*printf("rsp final: \n");
            //float *pA = (float *) (m_cfs[idx].df.data) + c * 2;
            float *pReszz = (float *) (resF.data);
            for (int h = 0; h < feaCF.rows; h++) {
                for (int w = 0; w < feaCF.cols; w++) {
                    pReszz[0] =  pReszz[0];

                    pReszz[1] =  pReszz[1];

                    printf("res:%f+%f i \t", pReszz[0], pReszz[1]);

                    pReszz += 2;

                }
                printf("\n");
            }
*/

       // denzz=denzz+FFTTools::real(FFTTools::fftd(resF, true));// by zhangzheng in 2017/5/24

       // float *Pdenzz = (float *)(denzz.data);

       // std::vector<cv::Mat> split12;
       // cv::split(denzz,split12);
      //  for(int i=0;i<denzz.rows;i++)
       // {
       //     for(int n=0;n< denzz.cols;n++)
       //     {
//
       //         for(int j=0;j< denzz.channels();j++) {
       //             printf("deb: %f \n", Pdenzz[0]);
       //             printf("%f \n", split12[0].at<float>(i,n));
       //             Pdenzz += 1;
       //         }
       //     }
       // }




   // return denzz;  // by zhangzheng in 2017/5/24
    return FFTTools::real(FFTTools::fftd(resF, true));
}

cv::Mat StapleTracker::detectTransPWP(int idx,
                                      cv::Mat &roiImg)
{
    cv::Mat res1 = cv::Mat::zeros(roiImg.rows,roiImg.cols,
                                 CV_32F);
    cv::Mat roiImg1= getSubWin(idx);
    cv::Mat res = cv::Mat::zeros(roiImg1.rows,roiImg1.cols,
                                  CV_32F);
    unsigned char *pImg = (unsigned char*)(roiImg1.data);
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
    float pos0=m_cfs[idx].pos[0];///m_cfs[idx].rate2img[0];   // code by zhangzheng in 2017/6/22
    float pos1=m_cfs[idx].pos[1];///m_cfs[idx].rate2img[1];

    float cx = m_cfs[idx].pos[0];
    float cy = m_cfs[idx].pos[1];
    //roi.width = round(m_cfs[idx].bg_size[0]);
    //roi.height = round(m_cfs[idx].bg_size[1]);
    //roi.width=round(m_cfs[idx].target_size[0]);
    //roi.height = round(m_cfs[idx].target_size[1]);
   // roi.width = floor(m_cfs[idx].window_sz[0]*m_cfs[idx].currentScaleFactor);
   // roi.height = floor(m_cfs[idx].window_sz[1]*m_cfs[idx].currentScaleFactor);
    roi.width = floor(m_cfs[idx].window_sz[0]*m_cfs[idx].scale);
    roi.height = floor(m_cfs[idx].window_sz[1]*m_cfs[idx].scale);
    //int tw = floor(m_cfs[idx].window_sz[0]*m_cfs[idx].scale);
    //int th = floor(m_cfs[idx].window_sz[1]*m_cfs[idx].scale);
    // roi.width=tw;/m_cfs[idx].rate2img[0];
    //roi.height=th;/m_cfs[idx].rate2img[1];
    roi.x = floor(cx - floor(roi.width/2.f)+1);
   roi.y = floor(cy - floor(roi.height/2.f)+1);
    int bg_w = round(m_cfs[idx].bg_size[0]*m_cfs[idx].scale);
    int bg_h = round(m_cfs[idx].bg_size[1]*m_cfs[idx].scale);

    // Get sub Image

    cv::Mat image = cv::Mat(m_img.height,m_img.width,
                            CV_8UC3, m_img.data[0]);

   // cvNamedWindow("test image size(640* 480 ??) ", 1);
   // cv::imshow("a",image);
   // cvWaitKey(100);
    if(roi.x<=0){roi.x=0;}
    if(roi.x>=image.rows){roi.x=image.rows;}
    if(roi.y>=image.cols){roi.y=image.cols;}
    if(roi.y<=0){roi.y=0;}
    cv::Mat ztemp = RectTools::subwindow(image, roi,
                                     cv::BORDER_REPLICATE);

   // cvNamedWindow("sf0,2");
  //  cv::imshow("2",ztemp);
   // cvWaitKey(100);
    cv::Mat z=cv::Mat::zeros(m_cfs[idx].window_sz[0], m_cfs[idx].window_sz[1],CV_32FC3);
    //if (z.cols != bg_w || z.rows != bg_h)
    //    cv::resize(z, z, cv::Size(bg_w, bg_h));
    if ((ztemp.cols != m_cfs[idx].window_sz[0]) || (ztemp.rows !=  m_cfs[idx].window_sz[1])) {
        if(m_cfs[idx].window_sz[0]>z.rows) {
       // cv::resize(ztemp, z, cv::Size(m_cfs[idx].window_sz[0], m_cfs[idx].window_sz[1]),0.0,0.0,1);
        cv::resize(ztemp, ztemp, cv::Size(m_cfs[idx].window_sz[0], m_cfs[idx].window_sz[1]),0.0,0.0,1);

          }
        else //if(m_cfs[idx].window_sz[0]<z.rows)
        {

        //
        //
        //
         cv::resize(ztemp, ztemp, cv::Size(m_cfs[idx].window_sz[0], m_cfs[idx].window_sz[1]),0.0,0.0,3);
        }
        return ztemp;
    }
    else{
        return ztemp;
    }
  //  cv::Mat roiGrayzz;
  //  cv::Mat imgGray;
  //  cv::cvtColor(z,roiGrayzz,CV_BGR2GRAY);
  ///  cv::cvtColor(image,imgGray,CV_BGR2GRAY);
  //  int *pzz= (int *)(z.data);
 //   int izz=0;
    /*printf("roiImg1: \n");
    for(int i=0;i<image.rows;i++)
    {
        for(int j=0;j<image.cols;j++)
        {
            for(int n=0;n<image.channels();n++)
            {
                if(n==0 && i>= roi.y && j>=roi.x) {
                    // printf("%d\t", z.data[i * roiGrayzz.cols + j]);
                    printf("%d\t", image.data[izz]);
                    //printf("%d\t", pzz[0]);
                }
                if(j==image.cols-1) printf("\n");
                izz++;
            }
        }
    }
    izz=0;
    printf("Img: \n");
    for(int i=0;i<ztemp.rows;i++)
    {
        for(int j=0;j<ztemp.cols;j++)
        {
            for(int n=0;n<ztemp.channels();n++)
            {
                if(n==0) {
                    // printf("%d\t", z.data[i * roiGrayzz.cols + j]);
                    printf("%d\t", ztemp.data[izz]);
                    //printf("%d\t", pzz[0]);
                }
                if(j==ztemp.cols-1) printf("\n");
                izz++;
            }
        }
    }
    printf("roiImg: \n");
    izz=0;
    for(int i=0;i<z.rows;i++)
    {
        for(int j=0;j<z.cols;j++)
        {
            for(int n=0;n<z.channels();n++)
            {
                if(n==0) {
                  // printf("%d\t", z.data[i * roiGrayzz.cols + j]);
                    printf("%d\t", z.data[i * z.cols *z.channels()+ j*z.channels() + n]);
                    //printf("%d\t", pzz[0]);
                }
                if(j==z.cols-1) printf("\n");
               // pzz++;
            }
        }
    }
    izz=0;
    printf("test by stupid zhangzheng\n");
    for(int i=0;i<imgGray.rows;i++)
    {
        for(int j=0;j<imgGray.cols;j++)
        {
            for(int n=0;n<imgGray.channels();n++)
            {
                if(n==0) {
                    // printf("%d\t", z.data[i * roiGrayzz.cols + j]);
                    printf("%d\t", imgGray.data[izz]);
                    //printf("%d\t", pzz[0]);
                }
                if(j==imgGray.cols-1) printf("\n");
                // pzz++;
                izz++;
            }
        }
    }

    //printf("test begin Y^_^Y!!!!___________________________________________________\n");*/
  //  std::vector<cv::Mat> split1;
 //   cv::split(z,split1);
 //   int idzz=0;
    /*for(int i=0;i<roiGrayzz.rows;i++)
    {
        for(int j=0;j<roiGrayzz.cols;j++)
        {

             printf("%d\t", roiGrayzz.data[idzz]);
            idzz++;
        }
        printf("\n");
    }*/

   /* cvNamedWindow("img_patch",1);
    cv::imshow("img_patch",z);
    cvWaitKey(190);*/
    //resize(z, z, cv::Size(bg_w, bg_h));

}

int StapleTracker::getTransFeaCF(cv::Mat &roiImg1,
                                 cv::Mat &feaHog1)
{
    // Extract HOG Feature
    cv::Mat roiGray;
    cv::cvtColor(roiImg1, roiGray, CV_BGR2GRAY);
    cv::Mat feaHog = fhog(roiGray, m_trans_cell_size,9);
    int izz=0;
    /*for(int i=0;i<roiImg1.rows;i++)
    {
        for(int j=0;j<roiImg1.cols;j++)
        {
           // for(int n=0;n<roiImg.channels();n++)
            {
              //  if(n==2) {
                    // printf("%d\t", z.data[i * roiGrayzz.cols + j]);
                    //printf("%d\t", roiImg.data[izz]);
                    printf("%d\t", roiGray.data[izz]);
             //   }
                if(j==roiImg1.cols-1) printf("\n");
                // pzz++;
                izz++;
            }
        }
    }*/

    //float * p = (float *)(feaHog.data );
   // for(int i=0;i<feaHog.rows;i++)
   // {
   //     for(int j=0;j <feaHog.cols;j++)
   //     {
   //         printf("%f  \t",p[0]);
   //         if(j==feaHog.cols-1) printf("\n");
   //         for(int n=0;n<feaHog.channels();n++)
 //           {
 //               p++;
  //          }
  //      }
   // }

    std::vector<cv::Mat> splitzz,splitzz1;
    std::vector<cv::Mat> res;
    cv::split(feaHog,splitzz);
    for(int i=0;i<feaHog.channels();i++)
    {
        res.push_back(splitzz[i]);
    }

   izz=0;
    cv::Mat roiImg;
    float az= (float)(feaHog.rows)/(roiImg1.rows);
    float bz= (float)(feaHog.cols)/(roiImg1.cols);
    //cv::resize(roiImg1,roiImg,cv::Size(feaHog.rows,feaHog.cols));//,0.0,0.0,0);
    //cv::resize(roiImg1(cv::Rect(0,0,4*feaHog.rows,4*feaHog.cols)),roiImg,cv::Size(0,0),0.25,0.25,1);//,0.0,0.0,0);
    //cv::resize(roiImg1,roiImg,cv::Size(feaHog.rows,feaHog.cols),0,0,1);//,0.0,0.0,0);
    if(feaHog.cols>roiImg1.rows)
    {
        cv::resize(roiImg1,roiImg,cv::Size(feaHog.rows,feaHog.cols),0,0,1);
    }
    else
    {
        cv::resize(roiImg1,roiImg,cv::Size(feaHog.rows,feaHog.cols),0,0,3);
    }
    /*for(int i=0;i<roiImg.rows;i++)
    {
        for(int j=0;j<roiImg.cols;j++)
        {
             for(int n=0;n<roiImg.channels();n++)
            {
                if(n==0) {
                // printf("%d\t", z.data[i * roiGrayzz.cols + j]);
                printf("%d\t", roiImg.data[izz]);
                //printf("%d\t", roiGray.data[izz]);
                   }
                if(j==roiImg.cols-1) printf("\n");
                // pzz++;
                izz++;
            }
        }
    }*/

    cv::split(roiImg,splitzz1);
    for(int i=0;i<roiImg.channels();i++)
    {
        res.push_back(splitzz1[i]);
    }

    int p2323=0;
    int p1212=0;
    int a=0;
   // int * p1 = (int *)(roiImg.data );

    //std::vector<cv::Mat> splitzz;
    //cv::split(roiImg.data,splitzz);
    //cv::merge(splitzz[2],splitzz[1],splitzz[0]);
    cv::Mat feaHogzz(roiImg.rows,roiImg.cols,CV_32FC(res.size()),cv::Scalar(0));
    float * pa=(float *)(feaHogzz.data);
    float * p = (float *)(feaHog.data );
  //  float * pr= (float *)(roiImg.data);
    std::vector<cv::Mat> spl;
    cv::split(roiImg,spl);
    //float * pfzz=(float *)();
    float maxzz=0.0;
    for(int i=0;i<roiImg.rows;i++)
    {
        for(int j=0;j <roiImg.cols;j++)
        {
            // printf("roiImg Channel 1:%f  \t",float(roiImg.data[(i*roiImg.cols+j)*(roiImg.channels())+0])/256);

            for(int n=0;n<roiImg.channels();n++)
            {
                //feaHogzz.data[a]=(((float)roiImg.data[p2323])/256)-0.5;
                pa[0]=(((float)roiImg.data[p2323])/255)-0.5;
                //pa[0]=(((float)roiImg.data[p2323])/255)-0.5;
               // pa[0]=(double)(pr[0])/255-0.5;
               // pa[0]=spl[n].at<float>(i,j);
                //pr++;
               /* if(n==0) {
                    printf("zz: %f \t ", pa[0]);
                }*/
                //printf("zz: %f \t",pa[0]);
                //printf("roiImg:%f\n",(((float)roiImg.data[p2323])/256)-0.5);
                //printf("roiImg:%f\n",(((float)roiImg.data[p2323])/256));
                /*if((((float)roiImg.data[p2323])/256)-0.5>maxzz) {maxzz=(((float)roiImg.data[p2323])/256)-0.5;}*/
                a++;
                // p1++;
                p2323++;
                pa++;

            }
            for(int j=0;j<feaHog.channels();j++)
            {
                //feaHogzz.data[a]=((float)feaHog.data[p1212]);
                //pa[0]=(((float)feaHog.data[p1212])/255)-0.5;
                //pa[0]=(((float)feaHog.data[p1212])/255)-0.5;
                pa[0]=p[0];
                p++;
                /*if(j==0) {
                    //printf("zz: %f \t ", pa[0]);
                }
               // printf("fuck off:feaHog: %f\n", ((float)feaHog.data[p1212])/256-0.5);
               // printf("fuck off:feaHog: %f\n", ((float)feaHog.data[p1212])/256);
                if((((float)feaHog.data[p1212])/256-0.5) > maxzz ) {maxzz= ((float)feaHog.data[p1212])/256-0.5;}*/
                a++;
                p1212++;
                pa++;

            }
            //if(j==roiImg.cols-1) printf("\n");
        }
    }
   // printf("\n");
   // printf("maxzz fea: %f \n",maxzz);
   // feaHog1=feaHogzz.clone();
    feaHog1=feaHogzz.clone();
    //return 0;
    p1212=0;
    p2323=0;
    float * pazz=(float *)(feaHog1.data);
    std::vector<cv::Mat> spfeaHog1;
    cv::split(feaHog1,spfeaHog1);
    /*for(int izz=0;izz<feaHog1.rows;izz++)
    {
        for(int jzz=0;jzz<feaHog1.cols;jzz++)
        {
            for(int nzz=0;nzz < feaHog1.channels();nzz++){
                if(nzz==0) {
                    if(jzz==0){
                        printf("feaCF before Hann:\t");
                    }
                    printf("%f \t", spfeaHog1[nzz].at<float>(izz, jzz));
                }
                pazz++;
            }
            if(jzz==feaHog1.cols-1) printf("\n");
        }
    }*/


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




int StapleTracker:: trainTransCF(int idx, cv::Mat &roiImg,
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

    int azz=0;

    std::vector<cv::Mat>feaCFhann;
    cv::split(m_cfs[idx].hann,feaCFhann);
    /*for(int izz=0;izz<m_cfs[idx].hann.rows;izz++)
    {
        for(int jzz=0;jzz<m_cfs[idx].hann.cols;jzz++)
        {
            //for(int nzz=0;nzz < m_cfs[idx].hann.channels();nzz++){
                //if(nzz==0) {
                    if(jzz==1 ){
                        printf("channels No.1: feaCF after Hann:\t");
                    }
                    printf("%f \t", feaCFhann[0].at<float>(izz, jzz));
               // }
                //feazz++;
           // }
            if(jzz==m_cfs[idx].hann.cols-1) printf("\n");
        }
    }*/






     azz=0;
    float * feazz=(float *)(feaCF.data);
    std::vector<cv::Mat>feaCF1;
    cv::split(feaCF,feaCF1);
   /* for(int izz=0;izz<feaCF.rows;izz++)
    {
        for(int jzz=0;jzz<feaCF.cols;jzz++)
        {
            for(int nzz=0;nzz < feaCF.channels();nzz++){
                if(nzz==0) {
                    if(jzz==1 ){
                        printf("channels No.1: feaCF after Hann:\t");
                    }
                    printf("%f + %f i\t", feaCF1[nzz].at<float>(izz, jzz),feaCF1[nzz+1].at<float>(izz, jzz));
                }
                feazz++;
            }
            if(jzz==feaCF.cols-1) printf("\n");
        }
    }*/


    //std::vector<cv::Mat> feaFFT;
    cv::Mat feaFFT;
    solveTransCF(num, den, numVZZ, ZX, feaCF,  m_cfs[idx].y,feaFFT,idx);


    m_cfs[idx].numVZZ=numVZZ.clone();

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
          //  printf("feaFFT.channels() : %d\n",feaFFT.channels());
            df1.push_back(cv::Mat::zeros(feaFFT.rows,feaFFT.cols,CV_32FC1));
            sf1.push_back(cv::Mat::zeros(feaFFT.rows,feaFFT.cols,CV_32FC1));
            Ldsf1.push_back(cv::Mat::zeros(feaFFT.rows,feaFFT.cols,CV_32FC1));
         }

        cv::merge(df1,m_cfs[idx].df);
        cv::merge(sf1,m_cfs[idx].sf);
        cv::merge(Ldsf1,m_cfs[idx].Ldsf);

        std::vector<cv::Mat> spllit1;
        cv::split(m_cfs[idx].df,spllit1);
        std::vector<cv::Mat> spllit2;
        cv::split(m_cfs[idx].sf,spllit2);
        std::vector<cv::Mat> spllit3;
        cv::split(m_cfs[idx].Ldsf,spllit3);

       // for(int iizz =0; iizz < m_cfs[idx].df.rows; iizz++)
       // {
       //     for(int jjzz=0; jjzz < m_cfs[idx].df.cols; jjzz++)
       //     {
       //         for(int nnzz=0; nnzz < floor((m_cfs[idx].df.channels())/2); nnzz++)
       //         {
       //             spllit1[nnzz*2].at<float>(iizz,jjzz)=0.0;
       //             spllit1[nnzz*2+1].at<float>(iizz,jjzz)=0.0;
       //         }
       //     }
       // }

       // for(int iizz =0; iizz < m_cfs[idx].sf.rows; iizz++)
       // {
        //    for(int jjzz=0; jjzz < m_cfs[idx].sf.cols; jjzz++)
        //    {
        //        for(int nnzz=0; nnzz < floor((m_cfs[idx].sf.channels())/2); nnzz++)
        //       {
        //           spllit2[nnzz*2].at<float>(iizz,jjzz)=0.0;
        //            spllit2[nnzz*2+1].at<float>(iizz,jjzz)=0.0;
        //       }
        //    }
        //}

        //for(int iizz =0; iizz < m_cfs[idx].Ldsf.rows; iizz++)
        //{
        //   for(int jjzz=0; jjzz < m_cfs[idx].Ldsf.cols; jjzz++)
        //    {
        //        for(int nnzz=0; nnzz < floor((m_cfs[idx].Ldsf.channels())/2); nnzz++)
        //        {
        //            spllit3[nnzz*2].at<float>(iizz,jjzz)=0.0;
        //            spllit3[nnzz*2+1].at<float>(iizz,jjzz)=0.0;
        //        }
        //    }
        //}





        //m_cfs[idx].numVZZ=numVZZ.clone();//numVZZ.clone();
        //m_cfs[idx].ZX=ZX.clone();//ZX.clone();




        float * p1= (float *)(m_cfs[idx].numVZZ.data);
        float * p2= (float *)(m_cfs[idx].ZX.data);
        std::vector<cv::Mat> split21;
        cv::split(m_cfs[idx].numVZZ,split21);
        std::vector<cv::Mat> split22;
        cv::split(m_cfs[idx].ZX,split22);
        std::vector<cv::Mat> split23;
        cv::split(m_cfs[idx].df,split23);
        std::vector<cv::Mat> split24;
        cv::split(m_cfs[idx].sf,split24);
        std::vector<cv::Mat> split25;
        cv::split(m_cfs[idx].Ldsf,split25);



       /* for(int i=0;i<numVZZ.rows;i++)
        {
            for(int j=0;j<numVZZ.cols;j++)
            {
                for(int n=0;n<floor(numVZZ.channels()/2);n++) {
                   // printf("conj(A).Mul A: %f + %fi // %f + %f i||||| conj(feaFFT)* Y: %f + %f i//// %f + %f i \n", p1[0],p1[1],split21[2*n].at<float>(i,j),split21[2*n+1].at<float>(i,j), p2[0], p2[1],split22[2*n].at<float>(i,j),split22[2*n+1].at<float>(i,j));
                    printf("m_df: %f + %f i\n",split23[n*2].at<float>(i,j), split23[n*2+1].at<float>(i,j));
                    printf("m_sf: %f + %f i\n",split24[n*2].at<float>(i,j),split24[n*2+1].at<float>(i,j));
                    printf("m_Ldsf: %f + %f i\n", split25[n*2].at<float>(i,j),split25[n*2+1].at<float>(i,j));
                    printf("m_numVZZ: %f + %f i\n", split21[n*2].at<float>(i,j),split21[n*2+1].at<float>(i,j));
                    p1 += 2;
                    p2 += 2;
                    //p3 += 2;
                }
            }
        }
*/

        m_cfs[idx].num = num.clone();
        m_cfs[idx].den = den.clone();
        m_cfs[idx].alpha = num.clone();
        m_cfs[idx].X=feaFFT.clone();
       // float * pfea = (float *)(fea.data);
        //float * pmcfss=(float *)(m_cfs[idx].numVZZ.data);
        //float * pmczx =(float *)(m_cfs[idx].ZX.data);
        //float * px = (float *)(m_cfs[idx].X.data);



       // int idxzz=0;
        //for(int i=0;i<numVZZ.rows;i++)
       // {
       //     for(int j=0;j<numVZZ.cols;j++)
        //    {
       //         if(j==numVZZ.cols-1) printf("\n ");
       //         for(int n=0;n<floor(numVZZ.channels()/2);n++) {
       //             pmcfss[0]=p1[0];
       //             pmcfss[1]=p1[1];
       //             pmczx[0]=p2[0];
       //             pmczx[1]=p2[1];
       //             printf(" m_cfs[idx].numVZZL %f + %f i [][]][]][][][] m_cfs[idx].ZX: %f + %f i+++++++++++m_cfs[idx].X: %f + %f i\t",pmcfss[0],pmcfss[1],pmczx[0],pmczx[1],px[0],px[1]);
       //             idxzz++;
       //             p1 += 2;
       //             p2 += 2;
       //             // pfeaFFT += 2;
       //  // pfea+=1;
       //         }
       //     }
       // }

      //  printf("\n");
        //by zz in 2017/5/18 init for some new values

       df1.clear();




        sf1.clear();
        Ldsf1.clear();


    }
    else
    {
        m_cfs[idx].num = (1-lr)*m_cfs[idx].num + lr*num;
        m_cfs[idx].den = (1-lr)*m_cfs[idx].den + lr*den;
        m_cfs[idx].den = (1-lr)*m_cfs[idx].den + lr*den;
        m_cfs[idx].numVZZ = (1-lr)*m_cfs[idx].numVZZ + lr*numVZZ;
        m_cfs[idx].ZX = (1-lr)*m_cfs[idx].ZX + lr*ZX;
        m_cfs[idx].X = (1-lr)*m_cfs[idx].X + lr*feaFFT;

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
        float *ptZZ= (float *)(numVZZ.data);
        float *ptZX= (float *)(ZX.data);
        float *pX= (float *)(m_cfs[idx].X.data);
        float *pfeaFFT = (float *)(feaFFT.data);

      /*  int channels =  feaCF.channels();
      //  for(int i=0;i<feasplit.size();i++)
        {
          for (int h=0; h<feasplit[0].rows; h++)
          {
              for (int w=0; w<feasplit[0].cols; w++)
              {
                  for (int c=0; c<floor(feasplit.size()/2); c++)
                  {
                      //if(pnumvZZ<(float *)(0x7fffe7a21000)) {
                         // printf("idx: %d ||||m_numVZZ: %f+%f !|||||m_den: %f+%f!||||m_ZX:%f+%f!|||numVZZ: %f+%f!||| ZX: %f+%f!",idx,pnumvZZ[0],pnumvZZ[1],pDen[0],pDen[1],pZX[0],pZX[1],ptZX[0],ptZX[1],ptZZ[0],ptZZ[1]);
                          pnumvZZ[0] = pnumvZZ[0] * (1 - BACF_lr) + BACF_lr * ptZZ[0];
                          pnumvZZ[1] = pnumvZZ[1] * (1 - BACF_lr) + BACF_lr * ptZZ[1];
                          pZX[0] = pZX[0] * (1 - BACF_lr) + BACF_lr * ptZX[0];
                          pZX[1] = pZX[1] * (1 - BACF_lr) + BACF_lr * ptZX[1];
                          pX[0] =pX[0]*(1-BACF_lr) + pfeaFFT[0]*BACF_lr;
                          pX[1] =pX[1]*(1-BACF_lr) + pfeaFFT[1]*BACF_lr;
                        //  printf("m_ZX: %f+%fi +++++m_ZZ: %f+%fi++++ m_X: %f+%fi \n",pZX[0],pZX[1],pnumvZZ[0],pnumvZZ[1],pX[0],pX[1]);
                      if(c==0)
                      {
                          //printf("zz:%f + %f i\t",pnumvZZ[0],pnumvZZ[1]);
                          printf("zx: %f + %f i\t",pZX[0],pZX[1]);
                          //printf("x :%f + %f i\t",pX[0],pX[1]);
                          if(w==feasplit[0].cols-1)
                          {
                              printf("\n");
                          }
                      }
                          pnumvZZ += 2;
                          ptZZ += 2;
                          pZX += 2;
                          ptZX += 2;
                          pX+=2;
                     // }
                  }
              }
          }
        }*/
        //delete(pnumvZZ);
       // delete(pDen);
       // delete(pZX);
       // delete(ptZX);
       // delete(ptZZ);

        feasplit.clear();
        feasplit1.clear();
        numVZZ.release();
        ZX.release();
        //by zz in 2017/5/18
    }

    //Compute the alpha
   // float *pA1 = (float *)(m_cfs[idx].alpha.data);
  //  float *pNum1 = (float *)(m_cfs[idx].num.data);
  //  float *pDen1 = (float *)(m_cfs[idx].den.data);
  //  int channels =  feaCF.channels();
  //  for(int iz1=0; iz1<feaCF.rows;iz1++)
  //  {
  //      for (int w=0; w<feaCF.cols; w++)
 //       {
 //           float factor = 1.0f/(*(pDen1++)+m_trans_lambda);
 //           for (int c=0; c<channels; c++)
 //           {
 //               pA1[0]=pNum1[0]*factor;
 //               pA1[1]=pNum1[1]*factor;
///
 //               printf("m_num: %f+%fi+++ m_alpha: %f+%fi\n",pNum1[0],pNum1[1],pA1[0],pA1[1]);
 //               pA1 += 2;
 //               pNum1 += 2;
 //           }
 //       }
 //   }


    int minItr = 1;
    int maxItr=2;
    int term = 1;
    int visfilt=0;
    //std::vector<cv::Mat> X=feaFFT;
    cv::Mat X=feaFFT;
    int  muo=1;
    //std::vector<cv::Mat> dfo,sfo,Ldsfo;
    int Mx0;
    int Mx1;
    Mx0=X.rows;
    Mx1=X.cols;
    int Nf=feaCF.channels();
    int Mf[2];
    Mf[0]=m_cfs[idx].bg_size[0];
    Mf[1]=m_cfs[idx].bg_size[1];
    ECF(idx,muo,X,Nf,term,minItr,maxItr,visfilt);
    X.release();
    feaFFT.release();
    //cv::Mat num, den, numVZZ, ZX,tempnumV,tempZX,tempZZ;
    num.release();
    den.release();

    tempnumV.release();
    tempZX.release();
    tempZZ.release();
    //delete(pA1);
    //delete(pNum1);
    //delete(pDen1);

    return 0;
}

bool StapleTracker::ECF(int idx, int muo, cv::Mat &X,int Nf,int term,int minItr,int maxItr,int visfilt) {
    //  int MMx=Mx[0]*Mx[1];
    int MMx = X.rows * X.cols;
    int MMx0 = X.rows;
    int MMx1 = X.cols;
    int Nx = Nf;
    float lambda = m_trans_lambda;
    int i = 1;
    m_cfs[idx].ADMM_iteration=2;
    m_cfs[idx].muo=muo;
    m_cfs[idx].beta=10;
    m_cfs[idx].mumax=1000;
    while (i <= m_cfs[idx].ADMM_iteration) {

        cv::Mat a1(1,1,CV_32FC2, cv::Scalar::all(0));
        //a2(0,0)[0]++;
        std::vector<cv::Mat> a2;
        cv::split(a1,a2);
        //cv::Mat <cv::Vec2f> a2=a1;
        cv::Mat b1=cv::Mat::zeros(1,1,CV_32FC2);
        cv::Mat c1=cv::Mat::zeros(1,1,CV_32FC2);
        //a1.data[0]=1.0;

        //a1.at<cv::Vec2f>(0,0)[0]=(float) 1.0;
        //a1.at<cv::Vec2f>(0,0)[1]=(float) 2.0;
        a2[0].at<float>(0,0)=-3;
        a2[1].at<float>(0,0)= -2;
        //a2(0,0)[0]++;
        //a1.at<cv::Vec2f>(0,0)[1]= (float) -2.0;

        //a1.data[1]=0-2.0;
       // b1.data[0]=-2.0;
        //b1.data[1]=4.0;

       // printf("a1:: %f,%f\n",a2[0].at<float>(0,0), a2[1].at<float>(0,0));
        std::cout<<a1.data[0]<<a1.data[1];

    //
        // c1=FFTTools::complexDivision(a1,b1);
       // c1.data[0]=(a1.data[0]*b1.data[0]+a1.data[1]*b1.data[1])/(b1.data[0]*b1.data[0]+b1.data[1]*b1.data[1]);
        //c1.data[1]=(a1.data[1]*b1.data[0]-a1.data[0]*b1.data[1])/(b1.data[0]*b1.data[0]+b1.data[1]*b1.data[1]);
       // printf("12312111312Y^_^Ydivision test result:  %f + %f i\n",c1.data[0],c1.data[1]);
        std::vector<cv::Mat> split1;
        cv::split(m_cfs[idx].df,split1);
        std::vector<cv::Mat> split2;
        cv::split(m_cfs[idx].numVZZ,split2);
        std::vector<cv::Mat> split3;
        cv::split(m_cfs[idx].ZX,split3);
        std::vector<cv::Mat> split4;
        cv::split(m_cfs[idx].Ldsf,split4);
        std::vector<cv::Mat> split5;
        cv::split(m_cfs[idx].sf,split5);

       /* for(int mzz=0;mzz < X.rows; mzz++)
        {
            for(int hzz=0; hzz< X.cols; hzz++)
            {
                for(int nzz=0; nzz< Nf; nzz++)
                {
                    printf("m_df: %f + %f i\n", split1[nzz*2].at<float>(mzz,hzz),split1[nzz*2+1].at<float>(mzz,hzz));
                    printf("m_sf: %f + %f i\n",split5[nzz*2].at<float>(mzz,hzz),split5[nzz*2+1].at<float>(mzz,hzz));
                    printf("m_Ldsf: %f + %f i\n",split4[nzz*2].at<float>(mzz,hzz),split4[nzz*2+1].at<float>(mzz,hzz));
                    printf("m_numVZZ: %f + %f i\n",split2[nzz*2].at<float>(mzz,hzz),split2[nzz*2+1].at<float>(mzz,hzz));

                }
            }
        }*/

        float * pasf = (float *)(m_cfs[idx].sf.data);

       // printf("%f,%d\n",m_cfs[idx].muo,muo);
        int izz1 = 0;
        for (int h = 0; h < X.rows; h++) {
            for (int w = 0; w < X.cols; w++) {
                for (int n = 0; n < Nf; n++) {
                    //by zz in 2017/5/20
                    //printf("%d,%d,%d\n", ++izz,Nf,m_cfs[idx].muo);


                   // printf("%f\n",split2[2*n].at<float>(h,w));
                    split2[2*n].at<float>(h,w) =  split2[2*n].at<float>(h,w)  + m_cfs[idx].muo;
                    if(n==0) {
                       // printf("%f  \t", split2[2 * n].at<float>(h, w));
                    }
                    //split2[2*n+1].at<float>(h,w) =  split2[2*n+1].at<float>(h,w)  + muo;


                    float aldsf1=m_cfs[idx].Ldsf.data[izz1];
                    split3[2*n].at<float>(h,w) =  split3[2*n].at<float>(h,w) + float(m_cfs[idx].muo) *  split1[2*n].at<float>(h,w)-  split4[2*n].at<float>(h,w);
                    split3[2*n+1].at<float>(h,w)= split3[2*n+1].at<float>(h,w) + float(m_cfs[idx].muo) * split1[2*n+1].at<float>(h,w) - split4[2*n+1].at<float>(h,w);

                    split5[2*n].at<float>(h,w) = ( split3[2*n].at<float>(h,w)* split2[2*n].at<float>(h,w)+ split3[2*n+1].at<float>(h,w)* split2[2*n+1].at<float>(h,w) )/ ( split2[2*n].at<float>(h,w)* split2[2*n].at<float>(h,w)+ split2[2*n+1].at<float>(h,w)* split2[2*n+1].at<float>(h,w)+0.00000000001);
                    float sfzz=m_cfs[idx].sf.data[izz1];

                    izz1++;
                    float aldsf2=m_cfs[idx].Ldsf.data[izz1];

                    split5[2*n+1].at<float>(h,w) = ( split3[2*n+1].at<float>(h,w)* split2[2*n].at<float>(h,w)- split3[2*n].at<float>(h,w)* split2[2*n+1].at<float>(h,w) )/ ( split2[2*n].at<float>(h,w)* split2[2*n].at<float>(h,w)+ split2[2*n+1].at<float>(h,w)* split2[2*n+1].at<float>(h,w)+0.00000000001);
                    float sfzz1= m_cfs[idx].sf.data[izz1];
                    //float
                    pasf[0]=split5[2*n].at<float>(h,w);
                    pasf[1]=split5[2*n+1].at<float>(h,w);
                    if(n==0) {
                        //printf("m_df: %f + %fi ////m_numVZZ:%f + %f i +++ m_ZX: %f + %f i  +++ m_Ldsf: %f + %f i ++++++m_sf: %f + %f i \n",
                        //       split1[2 * n].at<float>(h, w), split1[2 * n + 1].at<float>(h, w),
                        //       split2[2 * n].at<float>(h, w), split2[2 * n + 1].at<float>(h, w),
                        //       split3[2 * n].at<float>(h, w), split3[2 * n + 1].at<float>(h, w),
                        //       split4[2 * n].at<float>(h, w), split4[2 * n + 1].at<float>(h, w),
                        //       split5[2 * n].at<float>(h, w), split5[2 * n + 1].at<float>(h, w));

                        //printf("m_df: %f + %f i \t",split1[2 * n].at<float>(h, w), split1[2 * n + 1].at<float>(h, w));
                        //printf("m_numVZZ: %f + %f i \t",split2[2 * n].at<float>(h, w), split2[2 * n + 1].at<float>(h, w));
                        //printf("m_ZX: %f + %f i \t",split3[2 * n].at<float>(h, w), split3[2 * n + 1].at<float>(h, w));
                        //printf("m_Ldsf: %f + %f i \t",split4[2 * n].at<float>(h, w), split4[2 * n + 1].at<float>(h, w));
                       // printf("m_sf: %f + %f i \t",split5[2 * n].at<float>(h, w), split5[2 * n + 1].at<float>(h, w));

                    }
                    pasf+=2;
                    izz1++;
                    //numZ += 2;
                }
               // if(w==X.cols-1){printf("\n");}
            }
        }

        int N = Nf;
        int prodMx = MMx;

        //int M2 = std::floor(m_cfs[idx].target_size[0] * 0.3 *2);
        //int M1 = std::floor(m_cfs[idx].target_size[1] * 0.3 *2);
        int M2 = std::floor(m_cfs[idx].s_filt_sz[0]);
        int M1 = std::floor(m_cfs[idx].s_filt_sz[0]);
        float at = (m_cfs[idx].muo + lambda / (std::sqrt(prodMx)));

        izz1=0;
        float * pzzzdf= (float *)(m_cfs[idx].df.data);
        for (int h = 0; h < X.rows; h++) {
            for (int w = 0; w < X.cols; w++) {
                for (int n = 0; n < Nf; n++)
                {
                    split1[2*n].at<float>(h,w) = (split5[2*n].at<float>(h,w) * m_cfs[idx].muo + split4[2*n].at<float>(h,w) ) / at;
                    split1[2*n+1].at<float>(h,w)  = ( split5[1+2*n].at<float>(h,w)  * m_cfs[idx].muo + split4[2*n+1].at<float>(h,w)) / at;
                    if(n==0) {
                    //    printf("m_df: %f + %f i  \t", split1[2 * n].at<float>(h, w), split1[2 * n + 1].at<float>(h, w));
                    }
                    pzzzdf[0]=split1[2 * n].at<float>(h, w);
                    pzzzdf[1]=split1[2 * n + 1].at<float>(h, w);
                    pzzzdf+=2;
                    izz1+=2;
                }
              //  if(w==X.cols-1){printf("\n");}
            }
        }
        //test ifft through dft(inverse);
        cv::Mat a=cv::Mat(3,3,CV_32FC2);
        std::vector<cv::Mat> spa;
        cv::split(a,spa);
        cv::Mat b=cv::Mat(3,3,CV_32FC2);
        std::vector<cv::Mat> spb;
        //cv::split(b,spb);
      /*  for(int i=0;i<3;i++)
        {
            for(int j=0;j<3;j++)
            {
                spa[0].at<float>(i,j)=(float)(i*3+j+1);
                printf("%f",(float)(spa[0].at<float>(i,j)));
                spa[1].at<float>(i,j)=0;
            }
        }
        cv::merge(spa,a);
        //cv::idft(a,b,DFT_SCALE,0);
        b=FFTTools::fftd(a,true);
        cv::split(b,spb);
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<3;j++)
            {
                //spa[0].at<float>(i,j)=(float)(i*3+j);
                printf("%f + %f i\t",(float)(spb[0].at<float>(i,j)),(float)(spb[1].at<float>(i,j)));
                //spa[1].at<float>(i,j)=0;
            }
            printf("\n");
        }
*/
        //by zhangzheng in 2017 6.16
        cv::Mat xtem;
        std::vector<cv::Mat> xtempzz;
        std::vector<cv::Mat> dftemp;
        float *pdfz2=(float *)(m_cfs[idx].df.data);
        cv::split(m_cfs[idx].df,dftemp);
        for (int c=0; c<Nf; c++)
        {
            cv::Mat resF= cv::Mat::zeros(m_cfs[idx].df.rows, m_cfs[idx].df.cols,
                                         CV_32FC2);
            float *pRes3 = (float *)(resF.data);
            for (int h=0; h<m_cfs[idx].df.rows; h++)
            {
                for (int w=0; w<m_cfs[idx].df.cols; w++) {
                    pRes3[0]=dftemp[2*c].at<float>(h,w);
                    pRes3[1]=dftemp[2*c+1].at<float>(h,w);
                   /* if(c==0)
                   {
                       printf("xf: %f + %F i \t", pRes3[0],pRes3[1]);
                       if(w==m_cfs[idx].df.cols -1){printf("\n");}
                   }*/
                    pRes3 += 2;
                    pdfz2 += 2;
                }
            }
            xtempzz.push_back(FFTTools::fftd(resF, true));
        }
        cv::merge(xtempzz,xtem);
        std::vector<cv::Mat> splitx;
        cv::split(xtem,splitx);

        for(int mzz=0;mzz < xtem.rows; mzz++)
        {
            for(int hzz=0; hzz< xtem.cols; hzz++)
            {
                for(int nzz=0; nzz< splitx.size(); nzz++)
                {
                   /* if(nzz==0) {
                        printf("x1zz: %f + %f i \t", splitx[nzz * 2].at<float>(mzz, hzz),
                               splitx[nzz * 2 + 1].at<float>(mzz, hzz));
                    }*/
                }
              //  if(hzz==xtem.cols-1){printf("\n");}
            }
        }

        cv::Mat azztemp = m_cfs[idx].df.clone(); //A.t() transpose of A
        int delta[2];

        delta[0]=(M1-m_cfs[idx].df.rows>0)?std::floor((M1-m_cfs[idx].df.rows)/2):std::ceil((M1-m_cfs[idx].df.rows)/2);
        delta[1]=(M2-m_cfs[idx].df.cols>0)?std::floor((M2-m_cfs[idx].df.cols)/2):std::ceil((M2-m_cfs[idx].df.cols)/2);
        //azztemp=azztemp.t();
        //CvMat  matzz = azztemp;
        //cv::Mat matzz=cv::Mat(azztemp.rows,azztemp.cols,CV_32FC(azztemp.channels()));
         cv::Mat matzz=cv::Mat(azztemp.cols,azztemp.rows,CV_32FC(azztemp.channels()));
         //
        //cv::Mat timg =CvCreateImage();
        std::vector<cv::Mat> splitazz;
        //cv::split(m_cfs[idx].df,splitazz);
        cv::split(xtem,splitazz);
        std::vector<cv::Mat> splitmat;
        cv::split(matzz, splitmat);
        for(int tzz=0;tzz< splitmat.size();tzz++) {
            cv::transpose(splitazz[tzz], splitmat[tzz]);
        }
        //cv::merge(splitmat,matzz);
        cv::merge(splitazz,matzz);
        //float * pazz = (float *)(azztemp.data);
        std::vector<cv::Mat> splazz;
        cv::split(matzz,splazz);
        float * pmat = (float *)(matzz.data);
      /*  for(int izz=0;izz<azztemp.cols;izz++)
        {
            for(int jzz=0;jzz<azztemp.rows;jzz++)
            {
                for(int nzz=0;nzz<floor(azztemp.channels()/2);nzz++)
                {
                    if(nzz==0) {
                        printf("matzz: %f + %f i \t", splazz[nzz*2].at<float>(izz,jzz), splazz[nzz*2 +1].at<float>(izz,jzz));
                        //printf("matzz: %f + %f i \t", pmat[0], pmat[1]);
                        if(jzz==azztemp.rows-1){printf("\n");}
                    }
                    //pmat+=2;
                }
            }
        }*/

        cv::Mat matemp=cv::Mat(azztemp.rows,azztemp.cols, CV_32FC(azztemp.channels()));
        cv::Point2f P2f(delta[1],delta[0]);
        std::vector<cv::Mat> splpma1;
        cv::split(matzz,splpma1);
        std::vector<cv::Mat> splmatzz;
        cv::split(matemp,splmatzz);
        for(int i=0 ; i< splpma1.size();i ++)
        {
            shift(splpma1[i], splmatzz[i], P2f);
        }
        cv::merge(splmatzz,matemp);
        float * pmatemp= (float *)(matemp.data);
        std::vector<cv::Mat> splma;
        cv::split(matemp,splma);
       // printf("matemp:\n");
       /* for(int i=0;i<matemp.rows;i++)
        {
            for(int j=0;j<matemp.cols;j++)
            {
                for(int n=0;n<floor(matemp.channels()/2);n++)
                {
                    if(n==0) {
                        //printf("matemp: %f + %f i \t", pmatemp[0], pmatemp[1]);
                        printf("matemp: %f + %f i \t",splma[2*n].at<float>(i,j),splma[2*n+1].at<float>(i,j));
                        if(j==matemp.cols-1){printf("\n");}
                    }
                    pmatemp+=2;
                }
            }
        }*/


//CvMat * azz = matCircshift(CvMat * mat,int rowMove,int colMove);
       // CvMat * azz=cvCreateMat(matzz.rows,matzz.cols,CV_32FC(azztemp.channels()));
        //cv::cvConver t(matzz,azz);
        //cv::cvConvert(matzz,azz);
       // CvMat azz = matzz;
        cv::split(matemp,splitazz);
        cv::Mat r=cv::Mat(M1,M2,CV_32FC(matemp.channels()));
        r=matemp(cv::Rect(0,0,M2,M1));

        std::vector<cv::Mat> splitma;
        cv::split(m_cfs[idx].df,splitma);
       // for(int mzz=0;mzz< splitazz.size();mzz++) {
           // cv::transpose(splitazz[mzz], splitma[mzz]);
        //}

       // float * ptrump =(float *)(trump.data) ;
        std::vector<cv::Mat> splir;
        cv::split(r,splir);
        float * pmate = (float *)(r.data );
       /* for(int i=0 ; i< r.rows; i++) {
            for (int j = 0; j < r.cols; j++) {
                for( int n=0; n< r.channels(); n++)
                {
                    if(n==0) {
                        //printf("zztemp: %f + %f i \t", pmate[0], pmate[1]);
                        printf("r: %f + %f i \t", splir[2*n].at<float>(i,j), splir[2*n+1].at<float>(i,j));
                        if(j==r.cols-1){printf("\n");}
                    }
                   // pmate+=2;
                }
            }
        }*/

        cv::Mat zztemp;
        cv::merge(splitma,zztemp);


        int ch = Nf;
       // int stepCh = ch*2;
        std::vector<cv::Mat> xtemp1;
       // std::vector<cv::Mat> xfzz;
       // std::vector<cv::Mat>  sfzz;
        cv::Mat xtemp;
        //cv::split(m_cfs[idx].df,xtemp1);


      //  int delta[2];
      //  delta[0] = 0 - ((std::floor(M1 - MMx0) / 2) + 1) - 1;
      //  delta[1] = 0 - ((std::floor(M2 - MMx1) / 2) + 1) - 1;
      //  cv::Mat xf2 = cv::Mat::zeros(MMx0, MMx1, CV_32FC2);
      //  cv::Mat atemp;
//
      //  for (int n = 0; n < Nf; n++) {
      //      //cv::Mat r=numzz(cv::Rect(delta[0],delta[1],m_cfs[idx].fg_size[0],m_cfs[idx].fg_size[1]));
      //      xfzz.push_back(xf2);
      //  }
      //  xf2.release();
      //  cv::merge(xfzz,atemp);
      //  xfzz.clear();

        //atemp.clear();
       // cv::Mat sfo2 = RectTools::subwindow(xtemp,cv::Rect(delta[0], delta[1], M1, M2));
       // cv::Mat sfo2 = cv::Mat::zeros(M1,M2,CV_32FC(xtemp.channels()));
        //sfzz.push_back(sfo2);s
      // atemp(cv::Rect(delta[0], delta[1], M1, M2)) = sfo2;
        //atemp(cv::Rect(delta[0], delta[1], M1, M2))= 2.0;
        //sfzz=atemp;

        cv::Mat padded=cv::Mat::zeros(m_cfs[idx].df.rows,m_cfs[idx].df.cols,CV_32FC(m_cfs[idx].df.channels()));
        cv::split(padded,xtemp1);
       // padded.create(m_cfs[idx].df.rows,m_cfs[idx].df.cols,CV_32FC(m_cfs[idx].df.channels()));
      //  padded.setTo(cv::Scalar::all(0));
        //cv::Rect roi(cv::Point(floor((m_cfs[idx].df.cols-r.cols)/2),floor((m_cfs[idx].df.rows-r.rows)/2)),cv::Point(floor((m_cfs[idx].df.cols-r.cols)/2)+r.cols,floor((m_cfs[idx].df.rows-r.rows)/2)+r.rows));
        cv::Rect roi(floor((m_cfs[idx].df.cols-r.cols)/2),floor((m_cfs[idx].df.rows-r.rows)/2),r.cols,r.rows);
        //r.copyTo(padded(roi));
        //padded(cv::Rect(floor((m_cfs[idx].df.rows-r.rows)/2),floor((m_cfs[idx].df.cols-r.cols)/2),r.rows,r.cols)) = r;
        for(int i=0; i< r.channels();i++) {
           // xtemp1[i](cv::Rect(floor((m_cfs[idx].df.rows-r.rows)/2),floor((m_cfs[idx].df.cols-r.cols)/2),r.rows,r.cols))=splir[i];
            splir[i].copyTo(xtemp1[i](roi));
        }
        cv::merge(xtemp1,padded);
        //r.copyTo(padded);
        //printf("Y^_^Y yap\n");
        std::vector<cv::Mat> splipa;
        cv::split(padded,splipa);
       /* for(int i=0 ; i< padded.rows; i++) {
            for (int j = 0; j < padded.cols; j++) {
                for( int n=0; n< floor(splipa.size()/2); n++)
                {
                    if(n==0) {
                        printf("r: %f + %f i \t", splipa[2*n].at<float>(i,j), splipa[2*n+1].at<float>(i,j));
                        if(j==padded.cols-1){printf("\n");}
                    }
                }
            }
        }*/

        cv::Mat atemp=zztemp;

        //cv::split(resF.data)
        //cv::split();resF
        std::vector<cv::Mat> xtem3;
        float *pdf2=(float *)(m_cfs[idx].df.data);
        for (int c=0; c<ch; c++)
        {
            cv::Mat resF= cv::Mat::zeros(m_cfs[idx].df.rows, m_cfs[idx].df.cols,
                                    CV_32FC2);
                float *pRes3 = (float *)(resF.data);
                for (int h=0; h<m_cfs[idx].df.rows; h++)
                {
                    for (int w=0; w<m_cfs[idx].df.cols; w++) {
                        pRes3[0]=xtemp1[2*c].at<float>(h,w);
                        pRes3[1]=xtemp1[2*c+1].at<float>(h,w);
                        pdf2[0]=xtemp1[2*c].at<float>(h,w);
                        pdf2[1]=xtemp1[2*c+1].at<float>(h,w);
                       /* if(c==0){
                            printf("resF: %f + %f i \t",pdf2[0],pdf2[1]);
                            if(w==m_cfs[idx].df.cols-1){printf("\n");}
                        }*/
                           // printf("m_df: %f+%fi  +++ resF: %f+%fi \n",split1[2*c].at<float>(h,w),split1[2*c+1].at<float>(h,w),pRes3[0],pRes3[1]);
                        pRes3 += 2;
                        pdf2 += 2;
                    }
                }
            xtem3.push_back(FFTTools::fftd(resF));
        }
        cv::merge(xtem3,xtemp);
        xtemp1.clear();
        xtem3.clear();
        m_cfs[idx].df=xtemp.clone();

        std::vector<cv::Mat> splitxt;
        cv::split(m_cfs[idx].df,splitxt);
       // float * pxt= (float *)(xtemp.data);
        float * pmcf = (float *)(m_cfs[idx].df.data);
        for(int iz=0;iz<xtemp.rows;iz++)
        {
            for(int jz=0;jz<xtemp.cols;jz++)
            {
                for(int nz=0;nz< floor(xtemp.channels()/2);nz++)
                {
                   // printf("m_df2: %f + %f i\n",splitxt[2*nz].at<float>(iz,jz),splitxt[2*nz+1].at<float>(iz,jz));
                   // pmcf[0]=pxt[0];
                   // pmcf[1]=pxt[1];
                  /*  if(nz==0) {
                        printf("m_df2: %f + %f i\t", pmcf[0], pmcf[1]);
                        if(jz==xtemp.cols-1){printf("\n");}
                    }*/
                    pmcf+=2;
                   // pxt+=2;
                }
            }
        }








        cv::Mat resF11= cv::Mat::zeros(m_cfs[idx].df.rows, m_cfs[idx].df.cols,
                                       CV_32FC2);

        //cv::Mat resF11= cv::Mat::zeros(m_cfs[idx].df.rows, m_cfs[idx].df.cols,
        //                               CV_32FC2);
        std::vector<cv::Mat> split2331;
        cv::split(resF11,split2331);
        std::vector<cv::Mat> split2441;
        cv::split(atemp,split2441);
        cv::Mat tempzz;
      //  for (int c=0; c<ch; c++)
        //{
                //float *pA2 = (float *)((atemp.data)+c*2);
         //       //float *pRes2 = (float *)(resF11.data);
          //      for (int h=0; h<m_cfs[idx].df.rows; h++)
          //      {
         //           for (int w=0; w<m_cfs[idx].df.cols; w++) {
                       // if (pA2 < (float *) (0xab0000) && pA2 > (float *)(0x800051)) {

                            //pRes2[0] += pA2[0];
                            //pRes2[1] += pA2[1];
                            //split2331[0].at<float>(h,w)+=split2441[0+2*c].at<float>(h,w);
                            //split2331[1].at<float>(h,w)+=split2441[1+2*c].at<float>(h,w);

         //                   printf("atemp: %f+%fi+++ resF1: %f+%fi \n",split2331[0].at<float>(h,w),split2331[1].at<float>(h,w),split2441[0+2*c].at<float>(h,w),split2441[1+2*c].at<float>(h,w));

                   // }
        //              }
                //delete(pA2);
                //delete(pRes2);
         //       }
         //   cv::merge(split2441,tempzz);
        //    sfzz.push_back(FFTTools::fftd(tempzz, true));
        //}
        //cv::Mat dfo2 = (FFTTools::fftd(atemp, true));
        // dfozz.push_back(dfo2);
        //}


        //cv::Mat dfo2 = (FFTTools::fftd(atemp, true));
       // dfozz.push_back(dfo2);
      //}
       // cv::merge(sfzz,m_cfs[idx].df);
        atemp.release();
       // sfzz.clear();

       // std::vector<cv::Mat> split4;
       // cv::split(m_cfs[idx].df, split4);
        //float *pS4 = (float *) (m_cfs[idx].sf.data);
        //float *pL4 = (float *)(m_cfs[idx].Ldsf.data);
        //    float *pD4 = (float *) (m_cfs[idx].df.data);
            //    temp[n]= (m_cfs[idx].muo*m_cfs[idx].sf[n].data + m_cfs[idx].Ldsf[n].data)/at;
            //  }
        //    for (int h = 0; h < X.rows; h++)
        //    {
        //        for (int w = 0; w <X.cols; w++)
        //        {
        //            for (int n = 0; n < Nf; n++)
        //            {
        //            //by zz in 2017/5/20
        //            pS4[0] = pD4[0];
        //            pS4[1] = pD4[1];
        //
                    //  SFZ[0]=ZXZ[0]/numZ[0];
                    //  SFZ[1]=ZXZ[1]/numZ[1];
                   // pT += 2;
        //            pS4 += 2;
        //            pD4 += 2;

        //            }
        //        }

        //     }
       // delete(pS4);
       // delete(pD4);
       // split4.clear();

       // delete(pT2);
        //delete(pS2);
        //delete(pL2);
       // delete(numZ1);
      //  delete(p1);
       // delete(SFZ1);
       // delete(pdf1);
       // delete(ZXZ1);

            //cv::merge(sfzz,m_cfs[idx].sf);//
            // update Ldsf
         float *   pS5 = (float *) (m_cfs[idx].sf.data);
         float *   pL5 = (float *) (m_cfs[idx].Ldsf.data);
         float *   pD5 = (float *) (m_cfs[idx].df.data);



            for (int h = 0; h < X.rows; h++)
            {
                for (int w = 0; w < X.cols; w++)
                {
                    for (int n = 0; n < Nf; n++)
                    {
                        if(n==0) {
                            //printf("m_sf: %f+%fi+++ m_df: %f+%fi  +++ m_Ldsf: %f+%fi\n", pS5[0], pS5[1], pD5[0], pD5[1],
                            //       pL5[0], pL5[1]);
                            //printf("m_sf: %f + %f i \t", pS5[0], pS5[1]);
                            //printf("m_df: %f + %f i \t", pD5[0], pD5[1]);
                          //  printf("diff between sf and Ldsf %f + %f i \t",(pS5[0] - pD5[0]) * m_cfs[idx].muo,(pS5[1] - pD5[1]) * m_cfs[idx].muo);
                           // printf("m_Ldsf: %f + %f i \t", pL5[0], pL5[1]);
                           // if(w==X.cols-1){printf("\n");}
                        }
                        pL5[0] = pL5[0] + (pS5[0] - pD5[0]) * m_cfs[idx].muo;
                        pL5[1] = pL5[1] + (pS5[1] - pD5[1]) * m_cfs[idx].muo;
                        pD5 += 2;
                        pS5 += 2;
                        pL5 += 2;

                    }
                }
            }

            if((m_cfs[idx].muo * m_cfs[idx].beta) > m_cfs[idx].mumax) {
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
    if(offsetX<=0){ offsetX=0;}
    if(offsetY<=0){offsetY=0;}
    if(offsetY>=0.5*bg_w){offsetY=0.5*bg_w;}
    if(offsetX>=0.5*bg_h){offsetX=0.5*bg_h;}
    bgMask(cv::Rect(offsetX, offsetY,
                    bg_w-2*offsetX, bg_h-2*offsetY))=cv::Scalar::all(0.0);

    cv::Mat fgMask=cv::Mat::zeros(bg_h, bg_w, CV_8U);
    offsetX = (m_cfs[idx].bg_size[0]-m_cfs[idx].fg_size[0])*m_cfs[idx].scale/2;
    offsetY = (m_cfs[idx].bg_size[1]-m_cfs[idx].fg_size[1])*m_cfs[idx].scale/2;
    if(offsetX<=0){ offsetX=0;}
    if(offsetY<=0){offsetY=0;}
    if(offsetY>=0.5*bg_w){offsetY=0.5*bg_w;}
    if(offsetX>=0.5*bg_h){offsetX=0.5*bg_h;}
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
                                cv::Mat &fea, cv::Mat y, cv::Mat &feaFFT,int idx)
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

        numV.push_back(FFTTools::complexConjMult(y,feaFFT2));
       // float * p1= (float *)(feaFFT2.data);
        ///float * p2 = (float *)(y.data);
       // float * p3 = (float *)(feaSplit[i].data);
        //float * p3 = (float *)(fea.data);
       // for(int i=0;i<feaFFT2.rows;i++)
       // {
       //     for(int j=0;j<feaFFT2.cols;j++)
       //     {
        //        p1 += 2;
        //        for(int n=0;n<fea.channels();n++) {
       //  //           printf("y: %f+%f i ||||| feaFFT2: %f+%f i++++++fea: %f\n", p2[0], p2[1], p1[0], p1[1],p3[0]);

       //             p2 += 2;
        //            p3 += 1;
        //        }
        //    }
       // }

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
    numVZX.clear();
    feaFFT1.clear();


    //num = num*norm;
    //numVZZ=den;
    ZX=num;
    m_cfs[idx].numVZZ=numVZZ.clone();
    //m_cfs[idx].numVZZ=m_cfs[idx].numVZZ*norm;
    m_cfs[idx].ZX=ZX.clone();

    //printf("y: \n");
   int it=0;
    float *py=(float *)(y.data);
    float * pmZX= (float *)(m_cfs[idx].ZX.data);
   /* for(int i=0;i<y.rows;i++)
    {
        for(int j=0;j<y.cols;j++)
        {
            for(int n=0;n<y.channels();n++)
            { // if(n==0) {
                    printf("y: %f \t", py[0]);
                   // printf("y: %f \t",y.data[it] );
              //  }
                py+=1;
                //it++;
            }
           // if(j==y.cols-1){printf("\n");}
        }
    }*/

    float * p1= (float *)(numVZZ.data);
    float * p2= (float *)(ZX.data);
    float * pfeaFFT = (float *)(feaFFT.data);
    float * pfea = (float *)(fea.data);
    float * pmcfss=(float *)(m_cfs[idx].numVZZ.data);
    float * pmczx =(float *)(m_cfs[idx].ZX.data);
    std::vector<cv::Mat> split11;
    cv::split(m_cfs[idx].numVZZ,split11);
    std::vector<cv::Mat> split12;
    cv::split(m_cfs[idx].ZX,split12);
   // printf("numVZZL\n");
    int idxzz=0;
    for(int i=0;i<numVZZ.rows;i++)
    {
        for(int j=0;j<numVZZ.cols;j++)
        {

            for(int n=0;n<floor(numVZZ.channels()/2);n++) {
                //p1[0]=p1[0]*norm;
                pmcfss[0]=p1[0];
                pmcfss[1]=p1[1];
                pmczx[0]=p2[0];
                pmczx[1]=p2[1];
                //printf("|||| conj(feaFFT)* Y: %f+%f i||||||| feaFFT: %f+%fi  m_cfs[idx].numVZZL %f +%fi ,%f + %f i ______- m_cfsZX: %f + %f i\t", p2[0], p2[1],pfeaFFT[0],pfeaFFT[1],pmcfss[0],pmcfss[1], split11[n*2].at<float>(i,j),split11[n*2+1].at<float>(i,j),split12[n*2].at<float>(i,j),split12[n*2+1].at<float>(i,j));
                if(n==0) {
                   // printf("%f +%fi ,%f + %f i    \t", pmcfss[0], pmcfss[1], split11[n * 2].at<float>(i, j),
                    //       split11[n * 2 + 1].at<float>(i, j));
                }
                //if(n==0) {
                    //printf("feaFFT: %f + %f i\t",pfeaFFT[0],pfeaFFT[1]);
                  //  printf("ZZ: %f + %fi   \t", pmcfss[0], pmcfss[1]);
                    // printf("ZX: %f + %f i\t",p2[0],p2[1]);
                    // printf("fea: %f\t", pfea[0]);
                    // printf("feaFFT: %f + %f i\t", pfeaFFT[0],pfeaFFT[1]);
               // }
                idxzz++;
                p1 += 2;
                p2 += 2;
                pfeaFFT += 2;
                pfea+=1;
                pmcfss+=2; pmczx+=2;
            }
           // if(j==numVZZ.cols-1) printf(" \n ");
        }
    }
    cv::Mat b;
    CvMat a=b;
    m_cfs[idx].numVZZ=numVZZ.clone();
    split11.clear();
    split12.clear();
    //cv::resize;

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
       // printf("Y^_^Y");
        for (int w=0; w<width; w++)
        {
            float factor = *(pHann++);
            for (int c=0; c<channels; c++) {
                //printf("feadata:%f || hanndata: %f\n", pFea[c], factor);
                pFea[c] = pFea[c] * factor;
               // if(c==0) {
                   // printf("afnn: %f\t", pFea[c]);
                 //   if(w==width-1){printf("\n");}
                //}
              }
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
    //float output_sigma = std::sqrt((float)th*tw)*sigma/m_trans_cell_size;
    float output_sigma = std::sqrt((float)m_cfs[idx].s_filt_sz[0]*m_cfs[idx].s_filt_sz[1])*output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res.at<float>(i, j) = std::exp(mult* (ih*ih+jh*jh));
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
    int size = floor((res.cols + res.rows)/2 );
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

   // if (m_trans_cell_size > 1)
   //     cv::resize(newRes, newRes, cv::Size(m_cfs[idx].norm_delta_size,m_cfs[idx].norm_delta_size));
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
    int tw = round(m_cfs[idx].target_size1[0]);
    int th = round(m_cfs[idx].target_size1[1]);
    float factor = 1.f/(tw*th);

   // cv::Mat newRes=cv::Mat(cv::Size(m_cfs[idx].norm_delta_size,m_cfs[idx].norm_delta_size),
     //                      CV_32F);
    cv::Mat newRes=cv::Mat(cv::Size(0-(tw-res.rows)+1,0-(th-res.cols)+1),CV_32F);
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
       // printf("%f + %f i\t",ptr1[0],ptr1[1]);
        ptr1 += 2;
    }
   // printf("\n");
    int len=nScales;
    int minsz=std::min(len,nScalesInterp);

    float scaling=(float)nScalesInterp/len;
    int newSize=nScalesInterp;
    cv::Mat resH1 = cv::Mat(cv::Size(nScalesInterp, 1),
                           CV_32FC2);
    float * presH1=(float *)(resH1.data);
    float * presH= (float *)(resH.data);
    std::vector<cv::Mat> splitres;
   // cv::split(resH,splitres);
    int mids=ceil(minsz/2)+1;
    int mide=floor((minsz-1)/2)-1;

    //presH1=(float *)(resH1.data);
    //presH= (float *)(resH.data);
   // printf("test:\n");
    int a=0;
    for(int i=0;i<nScalesInterp;i++)
    {
        if(i<mids)
        {
            presH1[0]=scaling*presH[0];
            presH++;
            presH1[1]=scaling*presH[0];
            presH++;
        }
        else if((i<nScalesInterp-mide-1)&&(i>=mids))
        {
            presH1[0]=0;
            presH1[1]=0;
        }
        else //if(i>=nScalesInterp-mide-1)
        {
         //   presH1[0]=scaling*splitres[0].at<float>(i-(nScalesInterp-nScales),1);
           // presH1[1]=scaling*splitres[1].at<float>(i-(nScalesInterp-nScales),1);
            presH1[0]=scaling*presH[0];
            presH++;
            presH1[1]=scaling*presH[0];
            presH++;
        }
       // printf("test: %f + %f i \n",presH1[0],presH1[1]);
        presH1+=2;
        //presH+=2;
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

   // float tz_w, tz_h, bg_w, bg_h;//, fg_w, fg_h;
   // tz_w = m_cfs[idx].base_tz[0]*m_cfs[idx].scale_adapt;
   // tz_h = m_cfs[idx].base_tz[1]*m_cfs[idx].scale_adapt;
    //float pad = (tz_w+tz_h)/2.f;

  //  bg_w = tz_w+pad;
   // bg_h = tz_h+pad;
   // float scale = sqrt(m_trans_fixed_area/(bg_w*bg_h));
   // int  bg_area=round(sqrt(m_cfs[idx].target_size[0]*m_cfs[idx].target_size[1])*search_area_scale);
   // int pre_norm_bg_w=round(m_cfs[idx].bg_size[0]*m_cfs[idx].scale);
   // int pre_norm_bg_h=round(m_cfs[idx].bg_size[1]*m_cfs[idx].scale);
   // bg_w = pre_norm_bg_w/scale;
   // bg_h = pre_norm_bg_h/scale;
   // int fg_w = round(tz_w-pad*m_trans_inner_padding);
  //  int fg_h = round(tz_h-pad*m_trans_inner_padding);

    // Apply the new value
    //m_cfs[idx].target_size[0] = round(tz_w);
    //m_cfs[idx].target_size[1] = round(tz_h);
   // m_cfs[idx].target_size[0] = floor(m_cfs[idx].target_size1[0]*m_cfs[idx].currentScaleFactor);
   // m_cfs[idx].target_size[1] = floor(m_cfs[idx].target_size1[1]*m_cfs[idx].currentScaleFactor);
   // m_cfs[idx].bg_size[0] = bg_area-round((bg_area-m_cfs[idx].base_tz[0])%2);
   // m_cfs[idx].bg_size[1] = bg_area-(bg_area-m_cfs[idx].base_tz[1])%2;
  //  m_cfs[idx].fg_size[0] = (fg_w+((m_cfs[idx].bg_size[0])-fg_w)%2);
   // m_cfs[idx].fg_size[1] = fg_h+((m_cfs[idx].bg_size[1])-fg_w)%2;
  //  m_cfs[idx].currentScaleFactor = m_cfs[idx].scale_adapt * m_cfs[idx].currentScaleFactor;


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
   // m_cfs[idx].scale = scale;
    m_cfs[idx].scale =  m_cfs[idx].scale_adapt;

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

cv::Mat StapleTracker::getScaleFeaCF(int idx) {
   // int tw = m_cfs[idx].target_size1[0]*m_cfs[idx].currentScaleFactor;
   // int th = m_cfs[idx].target_size1[1]*m_cfs[idx].currentScaleFactor;
    int tw = m_cfs[idx].target_size1[0];
    int th = m_cfs[idx].target_size1[1];


    cv::Mat image = cv::Mat(m_img.height, m_img.width,
                            CV_8UC3, m_img.data[0]);


   // cv::Mat feaTmp, feas;
   // for (int i=0; i<nScales; i++)
   //// {
   //     int scale_tw = MAX_T(floor(tw*scale_factor[i]),m_scale_cell_size);
   //     int scale_th = MAX_T(floor(th*m_scale_factors[i]),m_scale_cell_size);

   //     cv::Rect roi;
   //     roi.width = scale_tw;
  // //     roi.height = scale_th;
   //     roi.x = round(m_cfs[idx].pos[0] - roi.width/2.f);
   //     roi.y = round(m_cfs[idx].pos[1] - roi.height/2.f);
    //    cv::Mat z;
   //     z = RectTools::subwindow(image, roi,
   //                              cv::BORDER_REPLICATE);
   //     if (z.cols != m_cfs[idx].scale_norm_tz[0] || z.rows != m_cfs[idx].scale_norm_tz[1])
  //          cv::resize(z,z,cv::Size(m_cfs[idx].scale_norm_tz[0],
 // /                                  m_cfs[idx].scale_norm_tz[1]));
  //      //resize(z,z,cv::Size(m_cfs[idx].scale_norm_tz[0],
  //      //                      m_cfs[idx].scale_norm_tz[1]));
  //      getOneScaleFeaCF(z, feaTmp);
  //      feaTmp = feaTmp.reshape(1,1);
   //     feaTmp = feaTmp * m_scale_hann[i];
 //       if (0==i)
 //           feas = feaTmp.clone();
 //       else
 //           feas.push_back(feaTmp);
 ///   }
 //   return feas;
    //float pos0=m_cfs[idx].pos[0]/m_cfs[idx].rate2img[0];
    //float pos1=m_cfs[idx].pos[1]/m_cfs[idx].rate2img[1];
    //float width0=tw/m_cfs[idx].rate2img[0];
    //float height0=th/m_cfs[idx].rate2img[1];
    cv::Mat feaTmp, feas;
    for (int i=0; i<m_scale_num; i++)
    {
        int scale_tw = MAX_T(floor(tw*m_scale_factors[i]),m_scale_cell_size);
        int scale_th = MAX_T(floor(th*m_scale_factors[i]),m_scale_cell_size);
        //int scale_tw = MAX_T(floor(width0*m_scale_factors[i]),m_scale_cell_size);
        //int scale_th = MAX_T(floor(height0*m_scale_factors[i]),m_scale_cell_size);

        cv::Rect roi;
        roi.width = scale_tw;
        roi.height = scale_th;

        roi.x = round(m_cfs[idx].pos[0] - roi.width/2.f);
        roi.y = round(m_cfs[idx].pos[1] - roi.height/2.f);
        //roi.x= round(pos0-roi.width/2.f);
        //roi.y = round(pos1 - roi.height/2.f);
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
