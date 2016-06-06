#ifndef __KCF_TRACKER_HPP__
#define __KCF_TRACKER_HPP__
#endif

#include "opencv2/opencv.hpp"
#include "tracker.hpp"

typedef struct tagKCFMats
{
    cv::Mat alphaf;
    cv::Mat num;
    cv::Mat den;
    cv::Mat prob;
    cv::Mat fea;
    cv::Mat hann;
    
    int feaDims[3]; //heightxwidthxfeaLen
    cv::Size tmpl_sz;
    float scale;
}KCFMats;

class KCFTracker : public Tracker
{
public:
    // Constructor
    KCFTracker(int maxSide, int minSide, int maxObjNum,
               bool isHog=true, bool isLAB=true,
               bool isFixedSize=true,
               bool isMultiScale=true);
    ~KCFTracker(){m_kcfs.clear();}

public:
    virtual int init();
    virtual int add(Rect_T &roi, int cate_id);
    virtual int update();    
    
protected:
    // Detect object in the current frame.
    cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value, int idx);
    
    // train tracker with a single image
    void train(cv::Mat fea, int idx, float lr);

    // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must also be periodic (ie., pre-processed with a cosine window).
    cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2, int idx);

    // Create Gaussian Peak. Function called only in the first frame.
    cv::Mat createGaussianPeak(int sizey, int sizex);
    
    // Obtain sub-window from image, with replication-padding and extract features
    cv::Mat getFeatures(int idx, float scale = 1.0f);

    int initHann(int idx);
    // Initialize Hanning window. Function called only in the first frame.
    cv::Mat createHanningMat(int height, int width, int feaLen);

    // Calculate sub-pixel peak for one dimension
    float subPixelPeak(float left, float center, float right);

private:
    // Trainning parameters
    float m_learn_rate; //Learning rate to update alpha
    float m_kcf_lambda; //regularization
    float m_kernel_gauss_sigma; //gaussian kernel bandwidth

    bool  m_isFixedSize;
    float m_y_gauss_sigma; // bandwidth of gaussian target
    float m_padding; // extra area surrounding the target
    int   m_template_size; // template size
    // For speed, (m_template_size/m_cell_size) should be a power of 2 or a product of small prime numbers.
    
    // Feature parameters
    bool  m_isHog;
    bool  m_isLAB; //complement the hog feature
    int   m_cell_size; //cell size for feature.
    cv::Mat m_LABCentroids;
    
    // Multi scale track parameter
    bool  m_isMultiScale;
    float m_scale_step; //scale step for multi-scale track
    float m_scale_weight; //Downweight track score of other scales for stability
    
    std::vector<KCFMats> m_kcfs;
};
