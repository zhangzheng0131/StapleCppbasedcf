#ifndef __KCF_EXP_TRACKER_HPP__
#define __KCF_EXP_TRACKER_HPP__

#include "opencv2/opencv.hpp"
#include "tracker.hpp"

//#define ENABLE_SUB_PEAK

typedef struct tagKCFExpMats
{
    //TransCF
    cv::Mat alpha;
    cv::Mat num;
    cv::Mat den;
    cv::Mat y;
    cv::Mat hann;
    cv::Mat fea;

    //TransPWP
    cv::Mat bgHist;
    cv::Mat fgHist;
    
    float pos[2];
    int target_size[2];
    int base_tz[2];
    float bg_size[2];
    float fg_size[2];
    int norm_delta_size;
    float scale;

    // ScaleCF
    cv::Mat num_scale;
    cv::Mat den_scale;
    int scale_norm_tz[2];
    float scale_max_factor;
    float scale_min_factor;
    float scale_adapt;
}KCFExpHandle;

class KCFExpTracker : public Tracker
{
public:
    // Constructor
    KCFExpTracker(int maxSide, int minSide, int maxObjNum);
    ~KCFExpTracker()
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
                     float lr, bool isInit=false);
    int solveTransCF(cv::Mat &num, cv::Mat &den,
                     cv::Mat &fea, cv::Mat y);
    
    int trainTransPWP(int idx,
                      cv::Mat &roiImg,
                      float lr, bool isInit=false);

    int detectTrans(int idx, float &conf);
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
    cv::Mat createGaussPeak(int idx, int sizey, int sizex, float sigma);
    cv::Mat gaussCorrelation(cv::Mat x1, cv::Mat x2);
    cv::Mat linearCorrelation(cv::Mat x1, cv::Mat x2);
    cv::Mat cropTransResponseCF(int idx, cv::Mat &res);
    cv::Mat cropTransResponsePWP(int idx, cv::Mat &res);
    
private:
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
    float m_trans_gauss_kernel_sigma;
    
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
    cv::Mat m_scale_y;
    std::vector<KCFExpHandle> m_cfs;
};

#endif
