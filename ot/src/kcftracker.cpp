#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"

// Constructor
KCFTracker::KCFTracker(int maxSide, int minSide,
                       int maxObjNum, bool isHog,
                       bool isLAB, bool isFixedSize,
                       bool isMultiScale)
    :Tracker(maxSide, minSide, maxObjNum)
{
    m_isHog = isHog;
    m_isLAB = isLAB;
    m_isFixedSize = isFixedSize;
    m_isMultiScale = isMultiScale;
    m_kcf_lambda = 0.0001;
    m_padding = 2.5;
    m_y_gauss_sigma = 0.125;
    
    if (m_isHog) {
        // TPAMI
        //m_learn_rate=0.02;
        //m_kernel_gauss_sigma=0.5; 
        m_learn_rate = 0.012;
        m_kernel_gauss_sigma = 0.6; 
        m_cell_size = 4;
        if (m_isLAB) {
            m_learn_rate = 0.005;
            m_kernel_gauss_sigma = 0.4;
            m_y_gauss_sigma = 0.1;
            m_LABCentroids = cv::Mat(nLABCentroid, 3,
                                     CV_32FC1,
                                     &pLABCentroids);
        }
    }
    else { // Gray pixel as feature
        m_learn_rate = 0.075;
        m_kernel_gauss_sigma = 0.2; 
        m_cell_size = 1;
    }
    
    if (m_isMultiScale)
    {
        m_template_size = 96;
        m_scale_step = 1.05;
        m_scale_weight = 0.95;
        m_isFixedSize = true;
    }
    else if (m_isFixedSize)
    {
        m_template_size = 96;
        m_scale_step = 1;
    }
    else
    {
        m_template_size = 1;
        m_scale_step = 1;
    }
    m_kcfs.resize(maxObjNum);
}

int KCFTracker::init()
{
    if (0!=Tracker::init())
        return -1;
    return 0;
}

// add object into tracker 
int KCFTracker::add(Rect_T &roi, int cate_id)
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
    m_objs[idx].roi = roi_s;
    m_objs[idx].cate_id = cate_id;
    m_objs[idx].obj_id = m_accumObjId++;
    m_objs[idx].status = 1;
    if (0 != initHann(idx))
        return -1;
    
    // extract feature
    m_kcfs[idx].fea = getFeatures(idx, 1);
    m_kcfs[idx].prob=createGaussianPeak(m_kcfs[idx].feaDims[0],
                                        m_kcfs[idx].feaDims[1]);
    m_kcfs[idx].alphaf=cv::Mat(m_kcfs[idx].feaDims[0],
                               m_kcfs[idx].feaDims[1],
                               CV_32FC2, float(0));
    m_kcfs[idx].num = cv::Mat(m_kcfs[idx].feaDims[0],
                              m_kcfs[idx].feaDims[1],
                              CV_32FC2, float(0));
    m_kcfs[idx].den = cv::Mat(m_kcfs[idx].feaDims[0],
                              m_kcfs[idx].feaDims[1],
                              CV_32FC2, float(0));
    train(m_kcfs[idx].fea, idx, 1.0);
    m_curObjNum ++;
    return 0;
 }

int KCFTracker::update()
{
    m_curObjNum = 0;
    for (int i=0 ;i<m_maxObjNum; i++)
    {
        if (-1 == m_objs[i].status)
            continue;
        Rect_T roi = m_objs[i].roi;
        if (roi.x+roi.w <= 0) roi.x = -roi.w + 1;
        if (roi.y+roi.h <= 0) roi.y = -roi.h + 1;
        if (roi.x >= m_img.width-1) roi.x=m_img.width-2;
        if (roi.y >= m_img.height-1) roi.y=m_img.height-2;
        m_objs[i].roi = roi;
        
        float cx = roi.x + roi.w / 2.0f;
        float cy = roi.y + roi.h / 2.0f;
        float peak_v;
        cv::Point2f res = detect(m_kcfs[i].fea,
                                 getFeatures(i, 1.0f),
                                 peak_v, i);
        if (m_isMultiScale)
        {
            float peak_v1, peak_v2;
            cv::Point2f res1, res2;
            res1 = detect(m_kcfs[i].fea,
                          getFeatures(i, 1.f/m_scale_step),
                          peak_v1, i);
            res2 = detect(m_kcfs[i].fea,
                          getFeatures(i, m_scale_step),
                          peak_v2, i);
            if (m_scale_weight*peak_v1>peak_v)
            {
                res = res1;
                peak_v = peak_v1;
                m_kcfs[i].scale /= m_scale_step;
                roi.w /= m_scale_step;
                roi.h /= m_scale_step;
            }
            else if (m_scale_weight*peak_v2>peak_v)
            {
                res = res2;
                peak_v = peak_v2;
                m_kcfs[i].scale *= m_scale_step;
                roi.w *= m_scale_step;
                roi.h *= m_scale_step;
            }
        }

        // Adjust by cell size and _scale
        roi.x = cx-roi.w/2.0f + ((float)res.x*m_cell_size * m_kcfs[i].scale);
        roi.y = cy-roi.h/2.0f + ((float)res.y*m_cell_size * m_kcfs[i].scale);
        
        if (roi.x >= m_img.width-1) roi.x=m_img.width-1;
        if (roi.y >= m_img.height-1) roi.y=m_img.height-1;
        if (roi.x+roi.w <= 0) roi.x = -roi.w + 2;
        if (roi.y+roi.h <= 0) roi.y = -roi.h + 2;
        m_objs[i].conf = peak_v;
        m_objs[i].roi = roi;
        
        cv::Mat x = getFeatures(i, 1);
        //if (peak_v>0.2f)
        train(x, i, m_learn_rate);
        m_curObjNum ++;
    }
    return m_curObjNum;
}

// Detect object in the current frame.
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value, int idx)
{
    using namespace FFTTools;
    cv::Mat k = gaussianCorrelation(x, z, idx);
    cv::Mat res = (real(fftd(complexMultiplication(m_kcfs[idx].alphaf, fftd(k)), true)));
    
    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
    peak_value = (float) pv;

    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);
    if (pi.x > 0 && pi.x < res.cols-1) {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x-1),
                            peak_value,
                            res.at<float>(pi.y, pi.x+1));
    }
    if (pi.y > 0 && pi.y < res.rows-1) {
        p.y += subPixelPeak(res.at<float>(pi.y-1, pi.x),
                            peak_value,
                            res.at<float>(pi.y+1, pi.x));
    }    
    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;
    return p;
}

void KCFTracker::train(cv::Mat fea, int idx, float lr)
{
    using namespace FFTTools;
#if 1
    cv::Mat k = gaussianCorrelation(fea, fea, idx);
    cv::Mat alphaf = complexDivision(m_kcfs[idx].prob,
                                     (fftd(k)+m_kcf_lambda));
    m_kcfs[idx].fea = (1-lr)*m_kcfs[idx].fea + lr*fea;
    m_kcfs[idx].alphaf = (1-lr)*m_kcfs[idx].alphaf+lr*alphaf;
#else
    cv::Mat kf = fftd(gaussianCorrelation(fea, fea, idx));
    cv::Mat num = complexMultiplication(kf, m_kcfs[idx].prob);
    cv::Mat den = complexMultiplication(kf, kf + m_kcf_lambda);
    
    m_kcfs[idx].fea = (1-lr)*m_kcfs[idx].fea + lr*fea;
    m_kcfs[idx].num = (1-lr)*m_kcfs[idx].num + lr*num;
    m_kcfs[idx].den = (1-lr)*m_kcfs[idx].den + lr*den;
    m_kcfs[idx].alphaf = complexDivision(m_kcfs[idx].num,
                                         m_kcfs[idx].den);
#endif
}

cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2, int idx)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(m_kcfs[idx].feaDims[1],
                                 m_kcfs[idx].feaDims[0]),
                        CV_32F, cv::Scalar(0) );
    // HOG features
    if (m_isHog)
    {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i=0; i<m_kcfs[idx].feaDims[2]; i++)
        {
            x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
            x1aux=x1aux.reshape(1, m_kcfs[idx].feaDims[0]);
            x2aux = x2.row(i).reshape(1, m_kcfs[idx].feaDims[0]);
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux),
                             caux, 0, true); 
            caux = fftd(caux, true);
            rearrange(caux);
            caux.convertTo(caux,CV_32F);
            c = c + real(caux);
        }
    }
    else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }
    cv::Mat d; 
    cv::max(( (cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0])- 2. * c) / (m_kcfs[idx].feaDims[0]*m_kcfs[idx].feaDims[1]*m_kcfs[idx].feaDims[2]) , 0, d);
    
    cv::Mat k;
    cv::exp((-d/(m_kernel_gauss_sigma*m_kernel_gauss_sigma)), k);
    return k;
}

cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);
    
    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;
    
    float output_sigma = std::sqrt((float) sizex * sizey) / m_padding * m_y_gauss_sigma;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
        }
    return FFTTools::fftd(res);
}

int KCFTracker::initHann(int idx)
{
    int padded_w = m_objs[idx].roi.w * m_padding;
    int padded_h = m_objs[idx].roi.h * m_padding;
    float scale;
    if (m_template_size > 1) {
        // Fit largest side to the given template size
        if (padded_w >= padded_h)  
            scale = padded_w/(float)m_template_size;
        else
            scale = padded_h/(float)m_template_size;
    }
    else {         
        if (padded_w*padded_h <= 10000) 
            scale = 1.f;
        else
            scale = 2.f;
    }
    int tmpl_w = padded_w / scale;
    int tmpl_h = padded_h / scale;
    
    if (m_cell_size>1) {
        int cell2 = 2*m_cell_size;
        tmpl_w = (int)((tmpl_w/cell2)*cell2) + cell2;
        tmpl_h = (int)((tmpl_h/cell2)*cell2) + cell2;
    }
    else {
        tmpl_w = (tmpl_w>>1)<<1;
        tmpl_h = (tmpl_h>>1)<<1;
    }
    m_kcfs[idx].tmpl_sz.width = tmpl_w;
    m_kcfs[idx].tmpl_sz.height = tmpl_h;
    m_kcfs[idx].scale = scale;
    if (m_isHog)
    {
        m_kcfs[idx].feaDims[0] = tmpl_h/m_cell_size - 2;
        m_kcfs[idx].feaDims[1] = tmpl_w/m_cell_size - 2;
        m_kcfs[idx].feaDims[2] = 31;
        if (m_isLAB)
            m_kcfs[idx].feaDims[2] += 15;
    }
    else
    {
        m_kcfs[idx].feaDims[0] = tmpl_h;
        m_kcfs[idx].feaDims[1] = tmpl_w;
        m_kcfs[idx].feaDims[2] = 1;
    }
    m_kcfs[idx].hann=createHanningMat(m_kcfs[idx].feaDims[0],
                                      m_kcfs[idx].feaDims[1],
                                      m_kcfs[idx].feaDims[2]);
    return 0;
}

cv::Mat KCFTracker::getFeatures(int idx, float scale)
{
    cv::Rect extracted_roi;
    float cx = m_objs[idx].roi.x + m_objs[idx].roi.w/2;
    float cy = m_objs[idx].roi.y + m_objs[idx].roi.h/2;
    
    extracted_roi.width=scale*m_kcfs[idx].scale*m_kcfs[idx].tmpl_sz.width;
    extracted_roi.height=scale*m_kcfs[idx].scale*m_kcfs[idx].tmpl_sz.height;

    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    cv::Mat FeaturesMap, image;
    image = cv::Mat(m_img.height,m_img.width,
                    CV_8UC3, m_img.data[0]);
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
    
    if (z.cols != m_kcfs[idx].tmpl_sz.width || z.rows != m_kcfs[idx].tmpl_sz.height) {
        cv::resize(z, z, m_kcfs[idx].tmpl_sz);
    }   
    
    if (m_isHog)
    {
        IplImage z_ipl = z;
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps(&z_ipl, m_cell_size, &map);
        normalizeAndTruncate(map, 0.2f);
        PCAFeatureMaps(map);
        FeaturesMap = cv::Mat(cv::Size(map->numFeatures,
                                       map->sizeX*map->sizeY),
                              CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();
        freeFeatureMapObject(&map);

        if (m_isLAB){
            int cell_sizeQ = m_cell_size*m_cell_size;
            cv::Mat imgLab;
            cvtColor(z, imgLab, CV_BGR2Lab);
            unsigned char *pLAB = (unsigned char*)(imgLab.data);
            cv::Mat LabFea = cv::Mat(m_LABCentroids.rows,
                                     m_kcfs[idx].feaDims[0]*m_kcfs[idx].feaDims[1],
                                     CV_32F, float(0));
            int cntCell = 0;
            for (int cY=m_cell_size; cY<z.rows-m_cell_size; cY+=m_cell_size){
                for (int cX=m_cell_size; cX<z.cols-m_cell_size; cX+=m_cell_size){
                    for(int y = cY; y < cY+m_cell_size; ++y){
                        for(int x = cX; x < cX+m_cell_size; ++x){
                            int idx = (z.cols * y + x) * 3;
                            float l = (float)pLAB[idx];
                            float a = (float)pLAB[idx + 1];
                            float b = (float)pLAB[idx + 2];
                            
                            // Iterate trough each centroid
                            float minDist = FLT_MAX;
                            int minIdx = 0;
                            float *inputCentroid=(float*)(m_LABCentroids.data);
                            for(int k=0; k<m_LABCentroids.rows; ++k){
                                float dist = ( (l - inputCentroid[3*k]) * (l - inputCentroid[3*k]) )
                                    + ( (a - inputCentroid[3*k+1]) * (a - inputCentroid[3*k+1]) ) 
                                    + ( (b - inputCentroid[3*k+2]) * (b - inputCentroid[3*k+2]) );
                                if(dist < minDist){
                                    minDist = dist;
                                    minIdx = k;
                                }
                            }
                            // Store result at output
                            LabFea.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ;
                        }
                    }
                    cntCell++;
                }
            }
            FeaturesMap.push_back(LabFea);
        }
    }
    else {
        FeaturesMap = RectTools::getGrayImage(z);
        FeaturesMap -= (float) 0.5; // In Paper;
    }
    
    return m_kcfs[idx].hann.mul(FeaturesMap);
}
    
cv::Mat KCFTracker::createHanningMat(int height,
                                     int width,
                                     int feaLen)
{   
    cv::Mat hann1t = cv::Mat(cv::Size(width,1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,height), CV_32F, cv::Scalar(0)); 
    
    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d=hann2t*hann1t, hann;
    if (feaLen>1) {
        cv::Mat hann1d = hann2d.reshape(1,1);
        hann = cv::Mat(cv::Size(height*width, feaLen),
                       CV_32F, cv::Scalar(0));
        for (int i = 0; i < feaLen; i++)
            for (int j = 0; j<height*width; j++) 
                hann.at<float>(i,j)=hann1d.at<float>(0,j);
    }
    else
        hann = hann2d;
    return hann;
}

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{   
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;
    
    return 0.5 * (right - left) / divisor;
}
