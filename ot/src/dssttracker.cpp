#include <math.h>
#include "dssttracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#include "comdef.h"
// Constructor
DSSTTracker::DSSTTracker(int maxSide, int minSide,
                       int maxObjNum, bool isHog,
                       bool isLAB, bool isFixedSize)
    :Tracker(maxSide, minSide, maxObjNum)
{
    m_isHog = isHog;
    m_isLAB = isLAB;
    m_isFixedSize = isFixedSize;
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
    m_isFixedSize = true;
    m_template_size = 96;

    m_scale_step = 1.02;
    m_scale_num  = 33;
    m_scale_sigma = 0.25;
    m_scale_factors = 0;
    m_scale_hann = 0;
    m_kcfs.resize(maxObjNum);
}

int DSSTTracker::init()
{
    if (0!=Tracker::init())
        return -1;

    if (0 != initScaleCF())
        return -1;
    return 0;
}

// add object into tracker 
int DSSTTracker::add(Rect_T &roi, int cate_id)
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

    trainScaleCF(idx, 1.0, true);
    return 0;
 }

int DSSTTracker::update()
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

        // Adjust by cell size and _scale
        roi.x = cx-roi.w/2.0f + ((float)res.x*m_cell_size * m_kcfs[i].scale);
        roi.y = cy-roi.h/2.0f + ((float)res.y*m_cell_size * m_kcfs[i].scale);
        
        if (roi.x >= m_img.width-1) roi.x=m_img.width-1;
        if (roi.y >= m_img.height-1) roi.y=m_img.height-1;
        if (roi.x+roi.w <= 0) roi.x = -roi.w + 2;
        if (roi.y+roi.h <= 0) roi.y = -roi.h + 2;
        m_objs[i].conf = peak_v;
        m_objs[i].roi = roi;

        //detect in scale
        if (1)
        {
            float bestScale;
            detectScaleCF(i, bestScale);
            cx = roi.x + roi.w / 2.0f;
            cy = roi.y + roi.h / 2.0f;
            roi.w = roi.w*bestScale;
            roi.h = roi.h*bestScale;
            roi.x = cx - roi.w/2.0f;
            roi.y = cy - roi.h/2.0f;
            if (roi.x >= m_img.width-1) roi.x=m_img.width-1;
            if (roi.y >= m_img.height-1) roi.y=m_img.height-1;
            if (roi.x+roi.w <= 0) roi.x = -roi.w + 2;
            if (roi.y+roi.h <= 0) roi.y = -roi.h + 2;
            m_objs[i].roi = roi;
            m_kcfs[i].scale_adapt *= bestScale;
            printf("best scale %f\n", bestScale);
            printf("cusr scale %f\n", m_kcfs[i].scale_adapt);
            trainScaleCF(i, 0.04);
        }
        // train translation CF
        cv::Mat x = getFeatures(i, 1);
        //if (peak_v>0.2f)
        train(x, i, m_learn_rate);
        m_curObjNum ++;
    }
    return m_curObjNum;
}

// Detect object in the current frame.
cv::Point2f DSSTTracker::detect(cv::Mat z, cv::Mat x, float &peak_value, int idx)
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

void DSSTTracker::train(cv::Mat fea, int idx, float lr)
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

cv::Mat DSSTTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2, int idx)
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

cv::Mat DSSTTracker::createGaussianPeak(int sizey, int sizex)
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

int DSSTTracker::initHann(int idx)
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
    m_kcfs[idx].scale_adapt = 1;
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

cv::Mat DSSTTracker::getFeatures(int idx, float scale)
{
    cv::Rect extracted_roi;
    float cx = m_objs[idx].roi.x + m_objs[idx].roi.w/2;
    float cy = m_objs[idx].roi.y + m_objs[idx].roi.h/2;
    
    
    extracted_roi.width=scale*m_kcfs[idx].scale*m_kcfs[idx].tmpl_sz.width/m_kcfs[idx].scale_adapt;
    extracted_roi.height=scale*m_kcfs[idx].scale*m_kcfs[idx].tmpl_sz.height/m_kcfs[idx].scale_adapt;

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
    
cv::Mat DSSTTracker::createHanningMat(int height,
                                     int width,
                                     int feaLen)
{   
    cv::Mat hann1t = cv::Mat(cv::Size(width,1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,height), CV_32F, cv::Scalar(0)); 
    
    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * PI_T * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * PI_T * i / (hann2t.rows - 1)));

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
float DSSTTracker::subPixelPeak(float left, float center, float right)
{   
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;
    
    return 0.5 * (right - left) / divisor;
}

int DSSTTracker::initScaleCF()
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
            m_scale_factors[i] = pow(m_scale_step, int(m_scale_num/2)-i);
    }
    
    cv::Mat y = cv::Mat(cv::Size(m_scale_num, 1),
                        CV_32F, cv::Scalar(0));
    
    float sigma = std::sqrt(m_scale_num)*m_scale_sigma;
    float mult = -0.5 / (sigma * sigma);
    
    for (int i = 0; i < m_scale_num; i++)
    {
        int dist = i-int(m_scale_num/2);
        y.at<float>(0, i)=std::exp(mult*dist*dist);
    }
    m_scale_y = FFTTools::fftd(y);
    return 0;
}

int DSSTTracker::detectScaleCF(int idx, float &bestScale)
{
    cv::Mat fea = getFeaScaleCF(idx);
    cv::Mat resH = cv::Mat(cv::Size(m_scale_num, 1),
                           CV_32FC2);
    float *ptr1, *ptr2, *ptr3, *ptr4;
    ptr1 = (float *)(resH.data);
    ptr2 = (float *)(fea.data);
    ptr3 = (float *)(m_kcfs[idx].num_scale.data);
    ptr4 = (float *)(m_kcfs[idx].den_scale.data);
    for (int i=0; i<m_scale_num; i++)
    {
        double realV=0, complV=0;
        for (int j=0; j<fea.cols; j++)
        {
            realV+=ptr2[2*j]*ptr3[2*j]-ptr2[2*j+1]*ptr3[2*j+1];
            complV+=ptr2[2*j]*ptr3[2*j+1]+ptr2[2*j+1]*ptr3[2*j];
        }
        ptr1[2*i] = realV / (ptr4[i] + 1e-3);;
        ptr1[2*i+1] = complV/ (ptr4[i] + 1e-3);;
        ptr2 += fea.step/4;
        ptr3 += m_kcfs[idx].num_scale.step/4;
    }

    cv::Mat response = FFTTools::real(FFTTools::fftd(resH, true));
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(response, NULL, &pv, NULL, &pi);
    bestScale = m_scale_factors[pi.x];
    return 0;
}

int DSSTTracker::trainScaleCF(int idx, float lr, bool isInit)
{
    cv::Mat fea = getFeaScaleCF(idx);
    if (isInit)
    {
        m_kcfs[idx].num_scale = cv::Mat(cv::Size(fea.cols,
                                                 fea.rows),
                                        CV_32FC2);
        m_kcfs[idx].den_scale = cv::Mat(cv::Size(m_scale_num,
                                                 1),
                                        CV_32F);
    }
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
            val+=ptr2[2*j]*ptr2[2*j]+ptr2[2*j+1]*ptr2[2*j+1];
        ptr1[i] = val;
        ptr2 += fea.step/4;
    }

    ptr1 = (float *)(num.data);
    ptr2 = (float *)(fea.data);
    ptr3 = (float *)(m_scale_y.data);
    for (int i=0; i<m_scale_num; i++)
    {
        for (int j=0; j<fea.cols; j++)
        {
            ptr1[j*2]=ptr2[2*j]*ptr3[0]+ptr2[2*j+1]*ptr3[1];
            ptr1[j*2+1]=ptr2[2*j]*ptr3[1]-ptr2[2*j+1]*ptr3[0];
        }
        ptr1 += num.step/4;
        ptr2 += fea.step/4;
        ptr3 += 2;
    }

    m_kcfs[idx].num_scale=(1-lr)*m_kcfs[idx].num_scale + lr*num;
    m_kcfs[idx].den_scale=(1-lr)*m_kcfs[idx].den_scale + lr*den;
    return 0;
}

cv::Mat DSSTTracker::getFeaScaleCF(int idx)
{
    cv::Rect roi;
    float cx = m_objs[idx].roi.x + m_objs[idx].roi.w/2;
    float cy = m_objs[idx].roi.y + m_objs[idx].roi.h/2;

    roi.width= m_scale_factors[0]*m_objs[idx].roi.w;
    roi.height= m_scale_factors[0]*m_objs[idx].roi.h;
    roi.x = cx - roi.width/2;
    roi.y = cy - roi.height/2;
    cv::Mat fea = getFeaOneScale(idx, roi,
                                 m_scale_hann[0]);

    for (int i=1; i<m_scale_num; i++)
    {
        roi.width= m_scale_factors[i]*m_objs[idx].roi.w;
        roi.height= m_scale_factors[i]*m_objs[idx].roi.h;
        roi.x = cx - roi.width/2;
        roi.y = cy - roi.height/2;
        fea.push_back(getFeaOneScale(idx, roi,
                                    m_scale_hann[i])); 
    }
    return FFTTools::fftd1d(fea, 1);    
}

cv::Mat DSSTTracker::getFeaOneScale(int idx, cv::Rect &roi,float hann)
{
    cv::Mat FeaturesMap, image;
    image = cv::Mat(m_img.height,m_img.width,
                    CV_8UC3, m_img.data[0]);
    cv::Mat z=RectTools::subwindow(image, roi,
                                   cv::BORDER_REPLICATE);
    
    //if (z.cols != m_kcfs[idx].tmpl_sz.width/2 || z.rows != m_kcfs[idx].tmpl_sz.height/2)
    //  cv::resize(z, z,
    //             cv::Size(m_kcfs[idx].tmpl_sz.width/2,
    //                      m_kcfs[idx].tmpl_sz.height/2));

    cv::resize(z, z,
               cv::Size(28,28));
                        
    //Extract HOG feature
    IplImage z_ipl = z;
    CvLSVMFeatureMapCaskade *map;
    getFeatureMaps(&z_ipl, m_cell_size, &map);
    normalizeAndTruncate(map, 0.2f);
    PCAFeatureMaps(map);
    FeaturesMap = cv::Mat(cv::Size(map->numFeatures,
                                   map->sizeX*map->sizeY),
                          CV_32F, map->map);
    FeaturesMap = FeaturesMap.reshape(1, 1);//map->numFeatures*map->sizeX*map->sizeY);
    freeFeatureMapObject(&map);
    FeaturesMap = hann*FeaturesMap;
    return FeaturesMap;
    //return FFTTools::fftd(FeaturesMap);
}
