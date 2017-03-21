// opencv2.4.13.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <time.h>
#include <stdio.h>
#include <ctype.h>
#include<iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include"wj.h"
using namespace std;
using namespace cv;
//不同的跟踪参数  
const double MHI_DURATION = 0.5;  
const double MAX_TIME_DELTA = 0.5;  
const double MIN_TIME_DELTA = 0.05;  
// 用于运动检测的循环帧数，与机器速度及FPS设置有关  
const int N = 2;  
IplImage **buf = 0;  
int last = 0;  
// 临时图像  
IplImage *mhi = 0; // MHI: 运动历史图像  
IplImage *orient = 0; // 方向  
IplImage *mask = 0; // 有效的运动掩码  
IplImage *segmask = 0; // 运动分割映射  
CvMemStorage* storage = 0; // 临时存储区  // parameters:  
//  img - input video frame  
//  dst - resultant motion picture  
//  args - optional parameters  
void   update_mhi( IplImage* img, IplImage* dst, int diff_threshold , IplImage* im){
	int countmotion = 0;
    double timestamp = (double)clock()/CLOCKS_PER_SEC; // 获取当前时间,以秒为单位  
    CvSize size = cvSize(img->width,img->height); // 获取当前帧尺寸  
    int i, idx1 = last, idx2;  
    IplImage* silh;  
    CvSeq* seq;  
    CvRect comp_rect;  
    double count;  
    double angle;  
    CvPoint center;  
    double magnitude;  
    CvScalar color;  

    // 开始时为图像分配内存 or 帧尺寸改变时重新分配内存  
    if( !mhi || mhi->width != size.width || mhi->height != size.height ){   //如果运动历史图像不符合要求进入    
        if( buf == 0 ){  
            buf = (IplImage**)malloc(N*sizeof(buf[0]));  
            memset( buf, 0, N*sizeof(buf[0]));  
        }  
        for( i = 0; i < N; i++ ){  

            cvReleaseImage( &buf[i] );  
            buf[i] = cvCreateImage( size, IPL_DEPTH_8U, 1 );  
            cvZero( buf[i] );  
        }  
        cvReleaseImage( &mhi );  
        cvReleaseImage( &orient );  
        cvReleaseImage( &segmask );  
        cvReleaseImage( &mask );  

        mhi = cvCreateImage( size, IPL_DEPTH_32F, 1 );  
        cvZero( mhi ); // clear MHI at the beginning  
        orient = cvCreateImage( size, IPL_DEPTH_32F, 1 );  
        segmask = cvCreateImage( size, IPL_DEPTH_32F, 1 );  
        mask = cvCreateImage( size, IPL_DEPTH_8U, 1 );  
    }  

    cvCvtColor( img, buf[last], CV_BGR2GRAY ); //RGB帧图像格式转换为gray  

    idx2 = (last + 1) % N; // index of (last - (N-1))th frame  
    last = idx2;  

    silh = buf[idx2];  
    // 相邻两帧的差  
    cvAbsDiff( buf[idx1], buf[idx2], silh );  
    cvThreshold( silh, silh, diff_threshold, 1, CV_THRESH_BINARY ); // 对差图像做二值化  
    cvUpdateMotionHistory( silh, mhi, timestamp, MHI_DURATION ); // 更新运动历史  
	/*函数只是更新 像素点的运动历史。也就说更新的不是图像，而是对图像中像素点运动情况的更新。
　　silh（x,y） !=0时，即该像素点发生运动，所以要对其进行更新，即mhi(x,y) = timestamp 表示运动发生的时刻
　　silh（x,y） =0时，即该像素点未发生运动，但还需检测对该点的跟踪时间是否超过了预设最大跟踪时间，即判断mhi(x,y)与timestamp -duration的大小。此时mhi(x,y)即为该点最近一次发生运动的时刻值，如其小于timestamp-duration,表示该点运动时刻已 经超出跟踪时间，故可以舍弃。
　　而当mhi(x,y)大于或者等于timestamp-duration时，表示该点此刻虽未发生运动，但还在跟踪时间内，所以不对该点发生运动的时间标记进行操作。
	*/
	im = cvCreateImage(cvGetSize(mhi), mhi->depth, mhi->nChannels);
	cvCopy(mhi, im, NULL);
    // convert MHI to blue 8u image  
    // cvCvtScale的第四个参数 shift = (MHI_DURATION - timestamp)*255./MHI_DURATION  
    // 控制帧差的消失速率  
    cvCvtScale( mhi, mask, 255./MHI_DURATION, (MHI_DURATION - timestamp)*255./MHI_DURATION );  
    cvZero( dst );  
    cvMerge( mask, 0, 0, 0, dst );

  // B,G,R,0 convert to BLUE image  

    // 计算运动的梯度方向以及正确的方向掩码  
    // Filter size = 3  
    cvCalcMotionGradient( mhi, mask, orient,  
                          MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );  
	//计算运动历史图像的梯度方向 (mask图像：标注运动梯度数据正确的点)
    if( !storage )  
        storage = cvCreateMemStorage(0);  
    else  
        cvClearMemStorage(storage);  

    // 运动分割： 获得运动部件的连续序列  
    seq = cvSegmentMotion( mhi, segmask, storage, timestamp, MAX_TIME_DELTA );  
	countmotion = 0;
    for( i = -1; i < seq->total; i++ ){  
        if( i < 0 ) {        // 对整幅图像操作  
            comp_rect = cvRect( 0, 0, size.width, size.height );  
            color = CV_RGB(255,255,255);  
            magnitude = 100;  // 画线长度以及圆半径的大小控制  
        }  
        else {          // 第i个运动组件  
            comp_rect = ((CvConnectedComp*)cvGetSeqElem( seq, i ))->rect;  
            // 去掉小的部分  
            if( comp_rect.width + comp_rect.height < 100 )  
                continue;  
            color = CV_RGB(255,0,0);  
            magnitude = 30;  
            //if(seq->total > 0) MessageBox(NULL,"Motion Detected",NULL,0);  
        }  
        // 选择组件ROI  
        cvSetImageROI( silh, comp_rect );   //在图像中设置兴趣区域 
        cvSetImageROI( mhi, comp_rect );  
        cvSetImageROI( orient, comp_rect );  
        cvSetImageROI( mask, comp_rect );  

        // 在选择的区域内，计算运动方向  
       angle = cvCalcGlobalOrientation( orient, mask, mhi, timestamp,MHI_DURATION);  
	   cout << angle << endl;
       angle = 360.0 - angle;  // adjust for images with top-left origin  
	   cout <<"angle = "<< angle <<"total="<< seq->total<<"i="<<i<< endl;
	   if (angle > 225.0&&angle < 315.0) {
		   countmotion++;
	   }
        // 在轮廓内计算点数  
        // Norm(L1) = 所有像素值的和  
        count = cvNorm( silh, 0, CV_L1, 0 );  
        cvResetImageROI( mhi );  
        cvResetImageROI( orient );  
        cvResetImageROI( mask );  
        cvResetImageROI( silh );  

        // 检查小运动的情形  
        if( count < comp_rect.width*comp_rect.height * 0.05 )  //  像素的5%  
            continue;  
        // 画一个带箭头的记录以表示方向  
		cout << "感兴趣的区域comp_recx.x=" << comp_rect.x << "  " << "comp_rect.y" << comp_rect.y << "  " << "comp_rect.width" << comp_rect.width << " " <<"comp_rect.height" << comp_rect.height << "  " <<endl;
		int rectwidth_mag = cvRound(comp_rect.width);
		int rectheight_mag = cvRound(comp_rect.height);
		center = cvPoint((comp_rect.x + comp_rect.width / 2), (comp_rect.y + comp_rect.height / 2));
		int mx0 = (comp_rect.x + comp_rect.width / 2);
		int my0 = (comp_rect.y + comp_rect.height / 2);
		int mx1 = cvRound(center.x + magnitude*cos(angle*CV_PI / 180));
		int my1 = cvRound(center.y - magnitude*sin(angle*CV_PI / 180));
        cvCircle( dst, center, cvRound(magnitude*1.2), color, 3, CV_AA, 0 );  
		cvCircle(dst, center,rectwidth_mag, color, 3, CV_AA, 0);
		//cvCircle(dst, center,rectheight_mag, color, 3, CV_AA, 0);
        cvLine( dst, center, cvPoint(mx1,my1),  color, 3, CV_AA, 0 );  
    } 
	if (countmotion > 5) {
		cout << "他好像是摔倒了"<<endl;
		cout << countmotion << endl;
		cin >> countmotion;
	}
}  
#define threshold_diff1 10 //设置简单帧差法阈值
#define threshold_diff2 10 //设置简单帧差法阈值
#define CV_CVX_WHITE    CV_RGB(0xff,0xff,0xff)
#define CV_CVX_BLACK    CV_RGB(0x00,0x00,0x00)
#define CHANNELS 3

IplImage *IavgF[NUM_CAMERAS], *IdiffF[NUM_CAMERAS], *IprevF[NUM_CAMERAS], *IhiF[NUM_CAMERAS], *IlowF[NUM_CAMERAS];
IplImage *Iscratch, *Iscratch2, *Igray1, *Igray2, *Igray3, *Imaskt;
IplImage *Ilow1[NUM_CAMERAS], *Ilow2[NUM_CAMERAS], *Ilow3[NUM_CAMERAS], *Ihi1[NUM_CAMERAS], *Ihi2[NUM_CAMERAS], *Ihi3[NUM_CAMERAS];

int CVCONTOUR_APPROX_LEVEL = 2;   // Approx.threshold - the bigger it is, the simpler is the boundary
int CVCLOSE_ITR = 1;
float Icount[NUM_CAMERAS];

void dian2(const cv::Mat & src)
{
	//assert(src.type() == CV_8UC3);
	cv::Mat dst;
	int width = 3 * src.cols;
	int height = src.rows;
	src.copyTo(dst);
	std::vector<uchar *> mFlag; //用于标记需要删除的点
								//对点标记
	for (int i = 0; i < height; ++i)
	{
		uchar * p = dst.ptr<uchar>(i);
		for (int j = 0; j < width; ++j)
		{
			uchar p1 = p[j];
			if (p1 != 1) continue;
			uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
			uchar p8 = (j == 0) ? 0 : *(p + j - 1);
			uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
			uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
			uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
			uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
			uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
			uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

			if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) == 1)
			{
				cout << "端点(" << i << "," << j << ")" << endl;
			}
		}
	}
}

cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1)
{
	cv::Mat dst;
	int width = 3 * src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //记录迭代次数
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达
			break;
		std::vector<uchar *> mFlag; //用于标记需要删除的点
									//对点标记
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记
				//  p9 p2 p3
				//  p8 p1 p4
				//  p7 p6 p5
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
					{
						//标记
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空
		}

		//对点标记
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记
				//  p9 p2 p3
				//  p8 p1 p4
				//  p7 p6 p5
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
					{
						//标记
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空
		}
	}
	return dst;
}

void delete_jut(Mat& src, Mat& dst, int uthreshold, int vthreshold, int type)
{
	//int threshold;
	src.copyTo(dst);
	int height = dst.rows;
	int width = dst.cols;
	int k;  //用于循环计数传递到外部
	for (int i = 0; i < height - 1; i++)
	{
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < width - 1; j++)
		{
			if (type == 0)
			{
				//行消除
				if (p[j] == 255 && p[j + 1] == 0)
				{
					if (j + uthreshold >= width)
					{
						for (int k = j + 1; k < width; k++)
							p[k] = 255;
					}
					else
					{
						for (k = j + 2; k <= j + uthreshold; k++)
						{
							if (p[k] == 255) break;
						}
						if (p[k] == 255)
						{
							for (int h = j; h <= k; h++)
								p[h] = 255;
						}
					}
				}
				//列消除
				if (p[j] == 255 && p[j + width] == 0)
				{
					if (i + vthreshold >= height)
					{
						for (k = j + width; k < j + (height - i)*width; k += width)
							p[k] = 255;
					}
					else
					{
						for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
						{
							if (p[k] == 255) break;
						}
						if (p[k] == 255)
						{
							for (int h = j; h <= k; h += width)
								p[h] = 255;
						}
					}
				}
			}
			else  //type = 1
			{
				//行消除
				if (p[j] == 0 && p[j + 1] == 255)
				{
					if (j + uthreshold >= width)
					{
						for (int k = j + 1; k < width; k++)
							p[k] = 0;
					}
					else
					{
						for (k = j + 2; k <= j + uthreshold; k++)
						{
							if (p[k] == 0) break;
						}
						if (p[k] == 0)
						{
							for (int h = j; h <= k; h++)
								p[h] = 0;
						}
					}
				}
				//列消除
				if (p[j] == 0 && p[j + width] == 255)
				{
					if (i + vthreshold >= height)
					{
						for (k = j + width; k < j + (height - i)*width; k += width)
							p[k] = 0;
					}
					else
					{
						for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
						{
							if (p[k] == 0) break;
						}
						if (p[k] == 0)
						{
							for (int h = j; h <= k; h += width)
								p[h] = 0;
						}
					}
				}
			}
		}
	}
}




void imageblur(Mat& src, Mat& dst, Size size, int threshold)
{
	int height = src.rows;
	int width = 3 * src.cols;
	blur(src, dst, size);
	for (int i = 0; i < height; i++)
	{
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < width; j++)
		{
			if (p[j] < threshold)
				p[j] = 0;
			else p[j] = 255;
		}
	}
}

void cvconnectedComponents(IplImage *mask, int poly1_hull0, float perimScale)
{
	static CvMemStorage*    mem_storage = NULL;
	static CvSeq*           contours = NULL;
	//CLEAN UP RAW MASK
	cvMorphologyEx(mask, mask, NULL, NULL, CV_MOP_OPEN, CVCLOSE_ITR);
	cvMorphologyEx(mask, mask, NULL, NULL, CV_MOP_OPEN, CVCLOSE_ITR);
	cvMorphologyEx(mask, mask, NULL, NULL, CV_MOP_OPEN, CVCLOSE_ITR);
	// 开运算先腐蚀在膨胀,腐蚀可以清除噪点,膨胀可以修复裂缝
	cvMorphologyEx(mask, mask, NULL, NULL, CV_MOP_CLOSE, CVCLOSE_ITR);
	cvMorphologyEx(mask, mask, NULL, NULL, CV_MOP_CLOSE, CVCLOSE_ITR);
	cvMorphologyEx(mask, mask, NULL, NULL, CV_MOP_CLOSE, CVCLOSE_ITR);
	// 闭运算先膨胀后腐蚀,之所以在开运算之后,因为噪点膨胀后再腐蚀,是不可能去除的
	//FIND CONTOURS AROUND ONLY BIGGER REGIONS
	if (mem_storage == NULL)
		mem_storage = cvCreateMemStorage(0);
	else cvClearMemStorage(mem_storage);
	//cvFindContours(mask,mem_storage,&contours,sizeof(CvContour),CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE,cvPoint(0,0) );
	CvContourScanner scanner = cvStartFindContours(mask, mem_storage, sizeof(CvContour), CV_RETR_EXTERNAL);
	//     在之前已讨论过这种轮廓方法,如果只提取轮廓没有cvFindContours()来的方便
	//     但下面要对轮廓进行直接操作,看来这种方法功能更强大一点
	CvSeq* c;
	int numCont = 0;
	while ((c = cvFindNextContour(scanner)) != NULL)
	{
		double len = cvContourPerimeter(c);
		// 计算轮廓周长
		double q = (mask->height + mask->width) / perimScale;
		// calculate perimeter len threshold
		if (len < q)
			// Get rid of blob if it's perimeter is too small
			cvSubstituteContour(scanner, NULL);
		else
			// Smooth it's edges if it's large enough
		{
			CvSeq* c_new;
			if (poly1_hull0)
				// Polygonal approximation of the segmentation
				c_new = cvApproxPoly(c, sizeof(CvContour), mem_storage,
					CV_POLY_APPROX_DP, CVCONTOUR_APPROX_LEVEL);
			else
				// Convex Hull of the segmentation
				c_new = cvConvexHull2(c, mem_storage, CV_CLOCKWISE, 1);

			cvSubstituteContour(scanner, c_new);
			// 替换轮廓扫描器中提取的轮廓
			numCont++;
			// 轮廓数目
		}
	}

	contours = cvEndFindContours(&scanner);
	// 结束扫描过程，并且返回最高层的第一个轮廓的指针

	// PAINT THE FOUND REGIONS BACK INTO THE IMAGE
	cvZero(mask);
	// 将掩模图像清零
	// 掩模: 指是由0和1组成的一个二进制图像

	//JUST DRAW PROCESSED CONTOURS INTO THE MASK
	for (c = contours; c != NULL; c = c->h_next)
		cvDrawContours(mask, c, CV_CVX_WHITE, CV_CVX_BLACK, -1, CV_FILLED, 8);
	//    return numCont;
}

void AllocateImages(IplImage *I)  //I is just a sample for allocation purposes
{
	for (int i = 0; i<NUM_CAMERAS; i++) {
		IavgF[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
		IdiffF[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
		IprevF[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
		IhiF[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
		IlowF[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
		Ilow1[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		Ilow2[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		Ilow3[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		Ihi1[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		Ihi2[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		Ihi3[i] = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
		cvZero(IavgF[i]);
		cvZero(IdiffF[i]);
		cvZero(IprevF[i]);
		cvZero(IhiF[i]);
		cvZero(IlowF[i]);
		Icount[i] = 0.00001; //Protect against divide by zero
	}
	Iscratch = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
	Iscratch2 = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3);
	Igray1 = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
	Igray2 = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
	Igray3 = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1);
	Imaskt = cvCreateImage(cvGetSize(I), IPL_DEPTH_8U, 1);
	cvZero(Iscratch);
	cvZero(Iscratch2);
}

void DeallocateImages()
{
	for (int i = 0; i<NUM_CAMERAS; i++) {
		cvReleaseImage(&IavgF[i]);
		cvReleaseImage(&IdiffF[i]);
		cvReleaseImage(&IprevF[i]);
		cvReleaseImage(&IhiF[i]);
		cvReleaseImage(&IlowF[i]);
		cvReleaseImage(&Ilow1[i]);
		cvReleaseImage(&Ilow2[i]);
		cvReleaseImage(&Ilow3[i]);
		cvReleaseImage(&Ihi1[i]);
		cvReleaseImage(&Ihi2[i]);
		cvReleaseImage(&Ihi3[i]);
	}
	cvReleaseImage(&Iscratch);
	cvReleaseImage(&Iscratch2);

	cvReleaseImage(&Igray1);
	cvReleaseImage(&Igray2);
	cvReleaseImage(&Igray3);

	cvReleaseImage(&Imaskt);
}

void accumulateBackground(IplImage *I, int number)
{
	static int first = 1;
	cvCvtScale(I, Iscratch, 1, 0); //To float;#define cvCvtScale cvConvertScale #define cvScale cvConvertScale
	if (!first) {
		cvAcc(Iscratch, IavgF[number]);//将2幅图像相加：IavgF[number]=IavgF[number]+Iscratch，IavgF[]里面装的是时间序列图片的累加
		cvAbsDiff(Iscratch, IprevF[number], Iscratch2);//将2幅图像相减：Iscratch2=abs(Iscratch-IprevF[number]);
		cvAcc(Iscratch2, IdiffF[number]);//IdiffF[]里面装的是图像差的累积和
		Icount[number] += 1.0;//累积的图片帧数计数
	}
	first = 0;
	cvCopy(Iscratch, IprevF[number]);//执行完该函数后，将当前帧数据保存为前一帧数据
}

void scaleHigh(float scale, int num)//设定背景建模时的高阈值函数
{
	cvConvertScale(IdiffF[num], Iscratch, scale); //Converts with rounding and saturation
	cvAdd(Iscratch, IavgF[num], IhiF[num]);//将平均累积图像与误差累积图像缩放scale倍然后再相加
	cvSplit(IhiF[num], Ihi1[num], Ihi2[num], Ihi3[num], 0);//#define cvCvtPixToPlane cvSplit,且cvSplit是将一个多通道矩阵转换为几个单通道矩阵
}

void scaleLow(float scale, int num)//设定背景建模时的低阈值函数
{
	cvConvertScale(IdiffF[num], Iscratch, scale); //Converts with rounding and saturation
	cvSub(IavgF[num], Iscratch, IlowF[num]);//将平均累积图像与误差累积图像缩放scale倍然后再相减
	cvSplit(IlowF[num], Ilow1[num], Ilow2[num], Ilow3[num], 0);
}

//Once you've learned the background long enough, turn it into a background model
void createModelsfromStats()
{
	for (int i = 0; i<NUM_CAMERAS; i++)
	{
		cvConvertScale(IavgF[i], IavgF[i], (double)(1.0 / Icount[i]));//此处为求出累积求和图像的平均值
		cvConvertScale(IdiffF[i], IdiffF[i], (double)(1.0 / Icount[i]));//此处为求出累计误差图像的平均值
		cvAddS(IdiffF[i], cvScalar(1.0, 1.0, 1.0), IdiffF[i]);  //Make sure diff is always something，cvAddS是用于一个数值和一个标量相加
		scaleHigh(0.7, i);//HIGH_SCALE_NUM初始定义为7，其实就是一个倍数
		scaleLow(0.6, i);//LOW_SCALE_NUM初始定义为6
	}
}

void backgroundDiff(IplImage *I, IplImage *Imask, int num)  //Mask should be grayscale
{
	cvCvtScale(I, Iscratch, 1, 0); //To float;
								   //Channel 1
	cvSplit(Iscratch, Igray1, Igray2, Igray3, 0);
	cvInRange(Igray1, Ilow1[num], Ihi1[num], Imask);//Igray1[]中相应的点在Ilow1[]和Ihi1[]之间时，Imask中相应的点为255(背景符合)
													//Channel 2
	cvInRange(Igray2, Ilow2[num], Ihi2[num], Imaskt);//也就是说对于每一幅图像的绝对值差小于绝对值差平均值的6倍或者大于绝对值差平均值的7倍被认为是前景图像
	cvOr(Imask, Imaskt, Imask);
	//Channel 3
	cvInRange(Igray3, Ilow3[num], Ihi3[num], Imaskt);//这里的固定阈值6和7太不合理了，还好工程后面可以根据实际情况手动调整！
	cvOr(Imask, Imaskt, Imask);
	//Finally, invert the results
	cvSubRS(Imask, cvScalar(255), Imask);//前景用255表示了，背景是用0表示
}
//of lengh = height*width
int maxMod[10];    //Add these (possibly negative) number onto max
				   // level when code_element determining if new pixel is foreground
int minMod[10];     //Subract these (possible negative) number from min
					//level code_element when determining if pixel is foreground
unsigned cbBounds[10]; //Code Book bounds for learning
bool ch[10];        //This sets what channels should be adjusted for background bounds
int nChannels = 10;
int imageLen = 0;
uchar *pColor; //YUV pointer

void nextpd(CvCapture *capture) {
	/*
	int i = 0;
	int startcapture = 1;
	int endcapture = 30;
	IplImage* pFrImg = NULL; //提取的前景图像，即运动目标
	IplImage* pBkImg = NULL; //背景图像
	IplImage *ImaskAVG = 0, *ImaskAVGCC = 0;
	IplImage* frame;//申请IplImage类型指针，就是申请内存空间来存放每一帧图像
	CvMat* pFrameMat = NULL; //原始视频矩阵
	CvMat* pFrMat = NULL;    //前景矩阵
	CvMat* pBkMat = NULL;    //背景矩阵
							 //创建窗口
	cvNamedWindow("video", 1);
	cvNamedWindow("background", 1);
	cvNamedWindow("foreground", 1);
	cvNamedWindow("AVG_ConnectComp", 1);//平均法连通区域分析后的图像
	cvNamedWindow("ForegroundAVG", 1);//平均法后图像
	maxMod[0] = 3;  //Set color thresholds to default values
	minMod[0] = 10;
	maxMod[1] = 1;
	minMod[1] = 1;
	maxMod[2] = 1;
	minMod[2] = 1;
	float scalehigh = 0.7;//默认值为6
	float scalelow = 0.6;//默认值为7
	while (1) {
		frame = cvQueryFrame(capture);// 从摄像头中抓取并返回每一帧
		if (!frame) break;//如果抓取帧为空   break 打破循环否则将抓取的那一帧显示在创建的窗口上
		char image_name[25];
		sprintf(image_name, "%s%d%s", "E:", ++i, ".jpg");//保存的图片名
		cvSaveImage(image_name, frame);   //保存一帧图片
		cvWaitKey(3);//延时，每秒钟约33帧；符合人眼观看速度；
		if (i == 1) {
		}
		else if (i == 2)
		{
			//AVG METHOD ALLOCATION
			AllocateImages(frame);//为算法的使用分配内存
			scaleHigh(scalehigh);//设定背景建模时的高阈值函数
			scaleLow(scalelow);//设定背景建模时的低阈值函数
			ImaskAVG = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
			ImaskAVGCC = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
			cvSet(ImaskAVG, cvScalar(255));
		}
		else
		{
			bool pause = false;
			bool singlestep = false;
			if (capture)
			{
				if (singlestep) {
					pause = true;
				}
				if (frame)
				{
					if (!pause && i >= startcapture && i < endcapture) {
						accumulateBackground(frame);//平均法累加过程
					}
					if (i == endcapture) {
						createModelsfromStats();//平均法建模过程
					}
					if (i >= endcapture) {//endcapture帧后开始检测前景
						backgroundDiff(frame, ImaskAVG);
						sprintf(image_name, "%s%d%s", "/Users/jiaoyukun/Downloads/opencv/", i, "(2).jpg");//保存的图片名
						cvSaveImage(image_name, ImaskAVGCC);   //保存一帧图片
						cvCopy(ImaskAVG, ImaskAVGCC);
						cvconnectedComponents(ImaskAVGCC, 1, 6.0);//平均法中的前景清除
						sprintf(image_name, "%s%d%s", "/Users/jiaoyukun/Downloads/opencv/", i, "(3).jpg");//保存的图片名
						cvSaveImage(image_name, ImaskAVGCC);   //保存一帧图片
						IplImage *imageresize = 0;
						imageresize = cvCreateImage(cvSize(512, 288), IPL_DEPTH_8U, 1);
						cvResize(ImaskAVGCC, imageresize, CV_INTER_LINEAR);
						cv::Mat src = cv::cvarrToMat(imageresize);
						delete_jut(src, src, 5, 5, 1);
						IplImage img = IplImage(src);
						cvMorphologyEx(&img, &img, NULL, NULL, CV_MOP_OPEN, 2);
						src = cv::cvarrToMat(&img);
						Size size;
						size.height = 10;
						size.width = 10;
						imageblur(src, src, size, 1);
						delete_jut(src, src, 5, 5, 0);
						img = IplImage(src);
						char image_name[25];
						sprintf(image_name, "%s%d%s", "/Users/jiaoyukun/Downloads/opencv/", i, "(4).jpg");//保存的图片名
						cvSaveImage(image_name, &img);   //保存一帧图片
						cv::Mat src2 = cv::cvarrToMat(&img);
						cv::threshold(src2, src2, 128, 1, cv::THRESH_BINARY);
						cv::Mat dst = thinImage(src2);//显示图像
						dian2(dst);
						dst = dst * 255;
						IplImage img2 = IplImage(dst);
						sprintf(image_name, "%s%d%s", "/Users/jiaoyukun/Downloads/opencv/", i, "(5).jpg");//保存的图片名
						cvSaveImage(image_name, &img2);   //保存一帧图片
														  //cvWaitKey(2);
					}
				}
			}
		}
	}
	//销毁窗口
	cvDestroyWindow("video");
	cvDestroyWindow("background");
	cvDestroyWindow("foreground");
	cvDestroyWindow("Example2");
	cvDestroyWindow("ForegroundAVG");
	cvDestroyWindow("AVG_ConnectComp");
	cvDestroyWindow("ForegroundCodeBook");
	cvDestroyWindow("CodeBook_ConnectComp");
	//释放图像和矩阵
	cvReleaseImage(&ImaskAVG);
	cvReleaseImage(&ImaskAVGCC);
	cvReleaseImage(&pFrImg);
	cvReleaseImage(&pBkImg);
	cvReleaseMat(&pFrameMat);
	cvReleaseMat(&pFrMat);
	cvReleaseMat(&pBkMat);
	DeallocateImages();//释放平均法背景建模过程中用到的内存
	cvReleaseCapture(&capture);//释放内存；
	*/
	return;
}
int main(int argc, char** argv){  

    IplImage* motion = 0;  
    CvCapture* capture = 0;
	IplImage* testmhi = 0;
	int s = 0;
	capture = cvCreateFileCapture("D:\\17.mp4");
	cvNamedWindow("mhi", 1);
    if( capture ){  

        cvNamedWindow( "Motion", 1 );  
        for(;;){  

           IplImage* image = cvQueryFrame( capture );
            if( !image )
                break;

            if( !motion )
            {
                motion = cvCreateImage( cvSize(image->width,image->height), 8, 3 );
                cvZero( motion );
                motion->origin = image->origin;
            }

            update_mhi( image, motion, 30 ,testmhi);
            cvShowImage( "Motion", motion );
			if (s == 1) {

			}
			cvShowImage("mhi", testmhi);
            if( cvWaitKey(10) >= 0 )
                break;
          }
        cvReleaseCapture( &capture );  
        cvDestroyWindow( "Motion" );  
    }  

    return 0;  
}
#ifdef _EiC
main(1,"motempl.c");
#endif