#include <algorithm>
#include <cmath>
#include <climits>
#include <vector>
#include <iostream>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#include <Eigen/Dense>

#include "Edge.h"
#include "FloatImage.h"
#include "Gaussian.h"
#include "GrayModel.h"
#include "GLine2D.h"
#include "GLineSegment2D.h"
#include "Gridder.h"
#include "Homography33.h"
#include "MathUtil.h"
#include "Quad.h"
#include "Segment.h"
#include "TagFamily.h"
#include "UnionFindSimple.h"
#include "XYWeight.h"

#include "TagDetector.h"

//#define DEBUG_APRIL

#ifdef DEBUG_APRIL
#include <opencv/cv.h>
#include <opencv/highgui.h>
#endif

using namespace std;

namespace AprilTags {

  std::vector<TagDetection> TagDetector::extractTags(const cv::Mat& image) {

    // Step 0: Optimized Pixel-to-Image Conversion using OpenCV float conversion
    int width = image.cols;
    int height = image.rows;
    
    cv::Mat imgFloat;
    image.convertTo(imgFloat, CV_32F, 1.0 / 255.0);
    
    AprilTags::FloatImage fimOrig(width, height);
    for (int y = 0; y < height; y++) {
      const float* srcRow = imgFloat.ptr<float>(y);
      for (int x = 0; x < width; x++) {
        fimOrig.set(x, y, srcRow[x]);
      }
    }

    std::pair<int,int> opticalCenter(width/2, height/2);

    //================================================================
    // Step one: preprocess image (convert to grayscale) and low pass if necessary

    FloatImage fim = fimOrig;
    
    float sigma = 0;
    float segSigma = 0.8f;

    if (sigma > 0) {
      int filtsz = ((int) max(3.0f, 3*sigma)) | 1;
      std::vector<float> filt = Gaussian::makeGaussianFilter(sigma, filtsz);
      fim.filterFactoredCentered(filt, filt);
    }

    //================================================================
    // Step two: Compute local gradient with SIMD (ARM NEON or AVX2) and fast atan2

    FloatImage fimSeg;
    if (segSigma > 0) {
      if (segSigma == sigma) {
        fimSeg = fim;
      } else {
        int filtsz = ((int) max(3.0f, 3*segSigma)) | 1;
        std::vector<float> filt = Gaussian::makeGaussianFilter(segSigma, filtsz);
        fimSeg = fimOrig;
        fimSeg.filterFactoredCentered(filt, filt);
      }
    } else {
      fimSeg = fimOrig;
    }

    FloatImage fimTheta(fimSeg.getWidth(), fimSeg.getHeight());
    FloatImage fimMag(fimSeg.getWidth(), fimSeg.getHeight());
    
    int widthSeg = fimSeg.getWidth();
    int heightSeg = fimSeg.getHeight();

    #pragma omp parallel for
    for (int y = 1; y < heightSeg - 1; y++) {
      int x = 1;
      int width_minus_1 = widthSeg - 1;

#if defined(__ARM_NEON)
      // Process 4 pixels at a time using ARM NEON
      for (; x <= width_minus_1 - 4; x += 4) {
        float left_vals[4] = { fimSeg.get(x-1, y), fimSeg.get(x, y), fimSeg.get(x+1, y), fimSeg.get(x+2, y) };
        float right_vals[4] = { fimSeg.get(x+1, y), fimSeg.get(x+2, y), fimSeg.get(x+3, y), fimSeg.get(x+4, y) };
        float32x4_t v_left = vld1q_f32(left_vals);
        float32x4_t v_right = vld1q_f32(right_vals);
        float32x4_t v_Ix = vsubq_f32(v_right, v_left);

        float up_vals[4] = { fimSeg.get(x, y-1), fimSeg.get(x+1, y-1), fimSeg.get(x+2, y-1), fimSeg.get(x+3, y-1) };
        float down_vals[4] = { fimSeg.get(x, y+1), fimSeg.get(x+1, y+1), fimSeg.get(x+2, y+1), fimSeg.get(x+3, y+1) };
        float32x4_t v_up = vld1q_f32(up_vals);
        float32x4_t v_down = vld1q_f32(down_vals);
        float32x4_t v_Iy = vsubq_f32(v_down, v_up);

        float32x4_t v_mag = vmlaq_f32(vmulq_f32(v_Ix, v_Ix), v_Iy, v_Iy);

        float ix_arr[4], iy_arr[4], mag_arr[4];
        vst1q_f32(ix_arr, v_Ix);
        vst1q_f32(iy_arr, v_Iy);
        vst1q_f32(mag_arr, v_mag);

        for (int k = 0; k < 4; k++) {
          float theta = MathUtil::fast_atan2(iy_arr[k], ix_arr[k]);
          fimTheta.set(x + k, y, theta);
          fimMag.set(x + k, y, mag_arr[k]);
        }
      }
#elif defined(__AVX2__)
      // Process 8 pixels at a time using AVX2
      for (; x <= width_minus_1 - 8; x += 8) {
        float left_vals[8] = { fimSeg.get(x-1, y), fimSeg.get(x, y), fimSeg.get(x+1, y), fimSeg.get(x+2, y),
                               fimSeg.get(x+3, y), fimSeg.get(x+4, y), fimSeg.get(x+5, y), fimSeg.get(x+6, y) };
        float right_vals[8] = { fimSeg.get(x+1, y), fimSeg.get(x+2, y), fimSeg.get(x+3, y), fimSeg.get(x+4, y),
                                fimSeg.get(x+5, y), fimSeg.get(x+6, y), fimSeg.get(x+7, y), fimSeg.get(x+8, y) };
        __m256 v_left = _mm256_loadu_ps(left_vals);
        __m256 v_right = _mm256_loadu_ps(right_vals);
        __m256 v_Ix = _mm256_sub_ps(v_right, v_left);

        float up_vals[8] = { fimSeg.get(x, y-1), fimSeg.get(x+1, y-1), fimSeg.get(x+2, y-1), fimSeg.get(x+3, y-1),
                             fimSeg.get(x+4, y-1), fimSeg.get(x+5, y-1), fimSeg.get(x+6, y-1), fimSeg.get(x+7, y-1) };
        float down_vals[8] = { fimSeg.get(x, y+1), fimSeg.get(x+1, y+1), fimSeg.get(x+2, y+1), fimSeg.get(x+3, y+1),
                               fimSeg.get(x+4, y+1), fimSeg.get(x+5, y+1), fimSeg.get(x+6, y+1), fimSeg.get(x+7, y+1) };
        __m256 v_up = _mm256_loadu_ps(up_vals);
        __m256 v_down = _mm256_loadu_ps(down_vals);
        __m256 v_Iy = _mm256_sub_ps(v_down, v_up);

        __m256 v_mag = _mm256_add_ps(_mm256_mul_ps(v_Ix, v_Ix), _mm256_mul_ps(v_Iy, v_Iy));

        float ix_arr[8], iy_arr[8], mag_arr[8];
        _mm256_storeu_ps(ix_arr, v_Ix);
        _mm256_storeu_ps(iy_arr, v_Iy);
        _mm256_storeu_ps(mag_arr, v_mag);

        for (int k = 0; k < 8; k++) {
          float theta = MathUtil::fast_atan2(iy_arr[k], ix_arr[k]);
          fimTheta.set(x + k, y, theta);
          fimMag.set(x + k, y, mag_arr[k]);
        }
      }
#endif

      // Scalar tail cleanup
      for (; x < width_minus_1; x++) {
        float Ix = fimSeg.get(x+1, y) - fimSeg.get(x-1, y);
        float Iy = fimSeg.get(x, y+1) - fimSeg.get(x, y-1);
        float mag = Ix*Ix + Iy*Iy;
        float theta = MathUtil::fast_atan2(Iy, Ix);
        
        fimTheta.set(x, y, theta);
        fimMag.set(x, y, mag);
      }
    }

    //================================================================
    // Step three: Extract edges by grouping pixels with similar thetas together.
    UnionFindSimple uf(fimSeg.getWidth()*fimSeg.getHeight());
    
    vector<Edge> edges(width*height*4);
    size_t nEdges = 0;

    { 
      vector<float> storage(width*height*4); 
      float * tmin = &storage[width*height*0];
      float * tmax = &storage[width*height*1];
      float * mmin = &storage[width*height*2];
      float * mmax = &storage[width*height*3];
                
      for (int y = 0; y+1 < height; y++) {
        for (int x = 0; x+1 < width; x++) {
          float mag0 = fimMag.get(x,y);
          if (mag0 < Edge::minMag)
            continue;
          mmax[y*width+x] = mag0;
          mmin[y*width+x] = mag0;
                    
          float theta0 = fimTheta.get(x,y);
          tmin[y*width+x] = theta0;
          tmax[y*width+x] = theta0;
                    
          Edge::calcEdges(theta0, x, y, fimTheta, fimMag, edges, nEdges);
        }
      }
                
      edges.resize(nEdges);
      std::stable_sort(edges.begin(), edges.end());
      Edge::mergeEdges(edges,uf,tmin,tmax,mmin,mmax);
    }
          
    //================================================================
    // Step four: Collect statistics for each cluster using a flat vector (replacing std::map)

    struct ClusterPoint {
      int rep;
      int x;
      int y;
      float mag;
      bool operator<(const ClusterPoint& other) const {
        return rep < other.rep;
      }
    };

    std::vector<ClusterPoint> clusterPoints;
    clusterPoints.reserve(width * height);

    for (int y = 0; y+1 < fimSeg.getHeight(); y++) {
      for (int x = 0; x+1 < fimSeg.getWidth(); x++) {
        int idx = y * fimSeg.getWidth() + x;
        if (uf.getSetSize(idx) < Segment::minimumSegmentSize)
          continue;

        int rep = (int) uf.getRepresentative(idx);
        clusterPoints.push_back({rep, x, y, fimMag.get(x, y)});
      }
    }

    std::sort(clusterPoints.begin(), clusterPoints.end());

    //================================================================
    // Step five: Loop over sorted flat clusters, fitting lines (Segments).
    std::vector<Segment> segments; 
    size_t nPoints = clusterPoints.size();
    
    for (size_t i = 0; i < nPoints; ) {
      int currentRep = clusterPoints[i].rep;
      size_t j = i;
      std::vector<XYWeight> points;
      while (j < nPoints && clusterPoints[j].rep == currentRep) {
        points.push_back(XYWeight(clusterPoints[j].x, clusterPoints[j].y, clusterPoints[j].mag));
        j++;
      }
      i = j;

      GLineSegment2D gseg = GLineSegment2D::lsqFitXYW(points);

      float length = MathUtil::distance2D(gseg.getP0(), gseg.getP1());
      if (length < Segment::minimumLineLength)
        continue;

      Segment seg;
      float dy = gseg.getP1().second - gseg.getP0().second;
      float dx = gseg.getP1().first - gseg.getP0().first;

      float tmpTheta = MathUtil::fast_atan2(dy, dx);

      seg.setTheta(tmpTheta);
      seg.setLength(length);

      float flip = 0, noflip = 0;
      for (unsigned int k = 0; k < points.size(); k++) {
        XYWeight xyw = points[k];
        float theta = fimTheta.get((int) xyw.x, (int) xyw.y);
        float mag = fimMag.get((int) xyw.x, (int) xyw.y);

        float err = MathUtil::mod2pi(theta - seg.getTheta());

        if (err < 0)
          noflip += mag;
        else
          flip += mag;
      }

      if (flip > noflip) {
        float temp = seg.getTheta() + (float)M_PI;
        seg.setTheta(temp);
      }

      float dot = dx*std::cos(seg.getTheta()) + dy*std::sin(seg.getTheta());
      if (dot > 0) {
        seg.setX0(gseg.getP1().first); seg.setY0(gseg.getP1().second);
        seg.setX1(gseg.getP0().first); seg.setY1(gseg.getP0().second);
      }
      else {
        seg.setX0(gseg.getP0().first); seg.setY0(gseg.getP0().second);
        seg.setX1(gseg.getP1().first); seg.setY1(gseg.getP1().second);
      }

      segments.push_back(seg);
    }

    //================================================================
    // Step six: Chain segments together using Gridder.
    Gridder<Segment> gridder(0,0,width,height,10);
    
    for (unsigned int i = 0; i < segments.size(); i++) {
      gridder.add(segments[i].getX0(), segments[i].getY0(), &segments[i]);
    }
    
    for (unsigned i = 0; i < segments.size(); i++) {
      Segment &parentseg = segments[i];
        
      GLine2D parentLine(std::pair<float,float>(parentseg.getX0(), parentseg.getY0()),
               std::pair<float,float>(parentseg.getX1(), parentseg.getY1()));

      Gridder<Segment>::iterator iter = gridder.find(parentseg.getX1(), parentseg.getY1(), 0.5f*parentseg.getLength());
      while(iter.hasNext()) {
        Segment &child = iter.next();
        if (MathUtil::mod2pi(child.getTheta() - parentseg.getTheta()) > 0) {
          continue;
        }

        GLine2D childLine(std::pair<float,float>(child.getX0(), child.getY0()),
                std::pair<float,float>(child.getX1(), child.getY1()));

        std::pair<float,float> p = parentLine.intersectionWith(childLine);
        if (p.first == -1) {
          continue;
        }

        float parentDist = MathUtil::distance2D(p, std::pair<float,float>(parentseg.getX1(),parentseg.getY1()));
        float childDist = MathUtil::distance2D(p, std::pair<float,float>(child.getX0(),child.getY0()));

        if (max(parentDist,childDist) > parentseg.getLength()) {
          continue;
        }

        parentseg.children.push_back(&child);
      }
    }

    //================================================================
    // Step seven: Search connected segments to form quads of length 4.
    vector<Quad> quads;
    vector<Segment*> tmp(5);
    for (unsigned int i = 0; i < segments.size(); i++) {
      tmp[0] = &segments[i];
      Quad::search(fimOrig, tmp, segments[i], 0, quads, opticalCenter);
    }

    //================================================================
    // Step eight: Decode the quads.
    std::vector<TagDetection> detections;

    for (unsigned int qi = 0; qi < quads.size(); qi++ ) {
      Quad &quad = quads[qi];

      GrayModel blackModel, whiteModel;
      const int dd = 2 * thisTagFamily.blackBorder + thisTagFamily.dimension;

      for (int iy = -1; iy <= dd; iy++) {
        float y = (iy + 0.5f) / dd;
        for (int ix = -1; ix <= dd; ix++) {
          float x = (ix + 0.5f) / dd;
          std::pair<float,float> pxy = quad.interpolate01(x, y);
          int irx = (int) (pxy.first + 0.5);
          int iry = (int) (pxy.second + 0.5);
          if (irx < 0 || irx >= width || iry < 0 || iry >= height)
            continue;
          float v = fim.get(irx, iry);
          if (iy == -1 || iy == dd || ix == -1 || ix == dd)
            whiteModel.addObservation(x, y, v);
          else if (iy == 0 || iy == (dd-1) || ix == 0 || ix == (dd-1))
            blackModel.addObservation(x, y, v);
        }
      }

      bool bad = false;
      unsigned long long tagCode = 0;
      for ( int iy = thisTagFamily.dimension-1; iy >= 0; iy-- ) {
        float y = (thisTagFamily.blackBorder + iy + 0.5f) / dd;
        for (int ix = 0; ix < thisTagFamily.dimension; ix++ ) {
          float x = (thisTagFamily.blackBorder + ix + 0.5f) / dd;
          std::pair<float,float> pxy = quad.interpolate01(x, y);
          int irx = (int) (pxy.first + 0.5);
          int iry = (int) (pxy.second + 0.5);
          if (irx < 0 || irx >= width || iry < 0 || iry >= height) {
            bad = true;
            continue;
          }
          float threshold = (blackModel.interpolate(x,y) + whiteModel.interpolate(x,y)) * 0.5f;
          float v = fim.get(irx, iry);
          tagCode = tagCode << 1;
          if ( v > threshold)
            tagCode |= 1;
        }
      }

      if ( !bad ) {
        TagDetection thisTagDetection;
        thisTagFamily.decode(thisTagDetection, tagCode);

        thisTagDetection.homography = quad.homography.getH();
        thisTagDetection.hxy = quad.homography.getCXY();

        float c = std::cos(thisTagDetection.rotation*(float)M_PI/2);
        float s = std::sin(thisTagDetection.rotation*(float)M_PI/2);
        Eigen::Matrix3d R;
        R.setZero();
        R(0,0) = R(1,1) = c;
        R(0,1) = -s;
        R(1,0) = s;
        R(2,2) = 1;
        Eigen::Matrix3d tmpMat;
        tmpMat = thisTagDetection.homography * R;
        thisTagDetection.homography = tmpMat;

        std::pair<float,float> bottomLeft = thisTagDetection.interpolate(-1,-1);
        int bestRot = -1;
        float bestDist = FLT_MAX;
        for ( int i=0; i<4; i++ ) {
          float const dist = AprilTags::MathUtil::distance2D(bottomLeft, quad.quadPoints[i]);
          if ( dist < bestDist ) {
            bestDist = dist;
            bestRot = i;
          }
        }

        for (int i=0; i< 4; i++)
          thisTagDetection.p[i] = quad.quadPoints[(i+bestRot) % 4];

        if (thisTagDetection.good) {
          thisTagDetection.cxy = quad.interpolate01(0.5f, 0.5f);
          thisTagDetection.observedPerimeter = quad.observedPerimeter;
          detections.push_back(thisTagDetection);
        }
      }
    }

    //================================================================
    // Step nine: Filter duplicate detections.
    std::vector<TagDetection> goodDetections;

    for ( vector<TagDetection>::const_iterator it = detections.begin();
        it != detections.end(); it++ ) {
      const TagDetection &thisTagDetection = *it;

      bool newFeature = true;

      for ( unsigned int odidx = 0; odidx < goodDetections.size(); odidx++) {
        TagDetection &otherTagDetection = goodDetections[odidx];

        if ( thisTagDetection.id != otherTagDetection.id ||
           ! thisTagDetection.overlapsTooMuch(otherTagDetection) )
          continue;

        newFeature = false;

        if ( thisTagDetection.hammingDistance > otherTagDetection.hammingDistance )
          continue;

        if ( thisTagDetection.hammingDistance < otherTagDetection.hammingDistance ||
           thisTagDetection.observedPerimeter > otherTagDetection.observedPerimeter )
          goodDetections[odidx] = thisTagDetection;
      }

       if ( newFeature )
         goodDetections.push_back(thisTagDetection);
    }

    return goodDetections;
  }

} // namespace
