#ifndef __STAPLE_TRACKER_HPP__
#define __STAPLE_TRACKER_HPP__

#include "opencv2/opencv.hpp"
#include "tracker.hpp"

#define ENABLE_SUB_PEAK
//#define ENABLE_LAB_TRANS
//#define ENABLE_LAB_SCALE
typedef struct tagStapleMats
{
    //TransCF

    cv::Mat alpha;
    cv::Mat num;
    cv::Mat den;
    cv::Mat y;
    cv::Mat hann;
    cv::Mat X;
    cv::Mat scale_basis;
    // by zz in 2017/05/16
    int s_filt_sz[2];
    int b_filt_sz[2];
    float rate2img[2];
    float term;
    int ADMM_iteration=2;
    //std::vector<cv::Mat>  df;
    ///std::vector<cv::Mat>  sf;
    //std::vector<cv::Mat>  Ldsf;
    //float *df;

    cv::Mat df;
    cv::Mat sf;
    cv::Mat Ldsf;
    cv::Mat ysf;
    cv::Mat firstImg;
    std::vector<cv::Mat>  P;
    cv::Mat ZX;
    cv::Mat numVZZ;
    std::vector<cv::Mat> feaFFT;
    cv::Mat imgor;
    cv::Mat s_num;
    cv::Mat sf_den;
    int muo;
////    int pfcell_size;
    int mumax;
    int beta;
    //TransPWP
    cv::Mat bgHist;
    cv::Mat fgHist;
    int last_target_sz[2];
    int last_pos[2];
    float pos[2];
    int target_size[2];
    float target_size1[2];
    float currentScaleFactor;
    int base_tz[2];
    int bg_size[2];
    int fg_size[2];
    int norm_delta_size;
    float scale;
    float window_sz[2];
//float pinterpScaleFactors[33];
//   float pinterpScaleFactors[33];

    // ScaleCF
    cv::Mat num_scale;
    cv::Mat den_scale;
    int scale_norm_tz[2];

   // int s_filt_sz[2];

    float scale_max_factor;
    float scale_min_factor;
    float scale_adapt;
}StapleHandle;

class StapleTracker : public Tracker
{
public:
    // Constructor
    StapleTracker(int maxSide, int minSide, int maxObjNum);
    ~StapleTracker()
    {
        m_cfs.clear();
        if (0 != m_scale_hann)
            delete []m_scale_hann;
        if (0 != m_scale_factors)
            delete []m_scale_factors;
    }

public:
    virtual int init();
    virtual int add(Rect_T &roi, int cate_id);
    virtual int update();

protected:
    int initTransCF();
    int initScaleCF();

    cv::Mat getSubWin(int idx);
    cv::Mat getSubWinPWP(int idx);

    int getTransFeaCF(cv::Mat &roiImg,
                      cv::Mat &feaHog);

    int trainTransCF(int idx,
                     cv::Mat &roiImg,
                     float lr, bool isInit=false, double BACF_lr=0.0125);
    int solveTransCF(cv::Mat &num, cv::Mat &den, cv::Mat &numVZZ, cv::Mat &ZX,
                     cv::Mat &fea, cv::Mat y, cv::Mat &feaFFT,int idx);

    int trainTransPWP(int idx,
                      cv::Mat &roiImg,
                      float lr, bool isInit=false);

    int detectTrans(int idx, float &conf);
    bool ECF(int idx, int muo, cv::Mat &X,int Nf,int term,int minItr,int maxItr,int visfilt);

    cv::Mat detectTransCF(int idx, cv::Mat &roiImg);
    cv::Mat detectTransPWP(int idx, cv::Mat &roiImg);
    Rect_T getShowRect(int idx);

    //Scale detection
    cv::Mat getScaleFeaCF(int idx);
    int getOneScaleFeaCF(cv::Mat &roiImg, cv::Mat &feaHog);
    int trainScaleCF(int idx, float lr, bool isInit=false);
    int detectScaleCF(int idx, float &conf);


private:
    cv::Mat createHann2D(int height, int width);
    int applyHann2D(cv::Mat& fea, cv::Mat& hann);
    cv::Mat createGaussianPeak(int idx, int sizey, int sizex, float sigma);
    cv::Mat cropTransResponseCF(int idx, cv::Mat &res);
    cv::Mat cropTransResponsePWP(int idx, cv::Mat &res);

private:
    // by zz in 2017/05/16 model paras

    //Feature paras
    int m_trans_cell_size;
    int m_trans_color_bins;
    float m_trans_inner_padding;
    float m_trans_fixed_area;
    float m_trans_lr_pwp;
    float m_trans_lr_cf;
    float m_trans_lambda;
    float m_trans_merge_factor;
    float m_trans_y_sigma;

    //Scale parameter
    int     m_scale_num;
    int     m_scale_cell_size;
    float   m_scale_max_area;
    float   m_scale_lr;
    float   m_scale_lambda;
    float   m_scale_step;
    float   m_scale_y_sigma;
    float  *m_scale_hann;
    float  *m_scale_factors;

    //by zhangzheng in 2017/5/23
    int frameH;
    int frameW;
    int visualization;
    int debug;
    int search_area_scale;
    float filter_size;
    float output_sigma_factor;
    int ini_imgs;
    float etha;// 0.0125;
    int upResize;
    int lowResize;
    float term;
    float lambda;
    float slambda;
//    int beta;
    int mu;
    int maxMu;
    char * search_area_shape;
    int min_image_sample_size;
    int max_image_sample_size;
    int fix_model_size;
    float pfcolorUpdataRate;
    char * pft[2];
    int pfhog_orientation;
    int pfnbin;
    int cell_size;
    cv::Mat pfw2c;
    struct pfcolorTransform{

    };
    float pfinterPatchRate;
    float currentScaleFactor;
    int pfgrey;
    int pfgreyHoG;
    int pfcolorProb;
    int pfcolorProbHoG;
    int pfcolorName;
    int pfgreyProb;
    int pflbp;
    int nScales;
    int nScalesInterp;
    float  scale_step;
    float scale_sigma_factor;
    int  scale_model_max_area;
    float scale_model_factor;
    float pscaleSizeFactors[17];
    char * s_num_compressed_dim;
    float interp_factor;
    float search_size[7];
    int resize_image;
    int resize_scale;
    int pfcell_size;
    float pbase_target_sz[2];//=(float)(target_sz[0]);
    int pfeatures_sz[2];//=b_filt_sz[0];
    int psz[2];//=sz[0];
    float pinterpScaleFactors[33];
    int pscale_model_sz[2];//=floor(m_cfs[idx].target_size[0]*scale_model_factor);
    //pscale_model_sz[1];//=floor(m_cfs[idx].target_size[1]*scale_model_factor);
    float pmin_scale_factor;//=pow(scale_step,ceil(log(max(5/psz[0], 5/psz[1])))/log(scale_step));
    float pmax_scale_factor;//=pow(scale_step,floor(log(min(roi.h/target_sz[0],roi.w/target_sz[1]))));
    int ps_num_compressed_dim;//=nScales;

    int MaxItr;
    int pmax_scale_dim;
    //
    float BACF_lr;
    cv::Mat m_scale_y;
    cv::Mat scale_window;
    std::vector<StapleHandle> m_cfs;
};

#endif
