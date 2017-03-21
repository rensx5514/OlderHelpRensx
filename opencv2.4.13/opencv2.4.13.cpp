// opencv2.4.13.cpp : �������̨Ӧ�ó������ڵ㡣
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
//��ͬ�ĸ��ٲ���  
const double MHI_DURATION = 0.5;  
const double MAX_TIME_DELTA = 0.5;  
const double MIN_TIME_DELTA = 0.05;  
// �����˶�����ѭ��֡����������ٶȼ�FPS�����й�  
const int N = 2;  
IplImage **buf = 0;  
int last = 0;  
// ��ʱͼ��  
IplImage *mhi = 0; // MHI: �˶���ʷͼ��  
IplImage *orient = 0; // ����  
IplImage *mask = 0; // ��Ч���˶�����  
IplImage *segmask = 0; // �˶��ָ�ӳ��  
CvMemStorage* storage = 0; // ��ʱ�洢��  // parameters:  
//  img - input video frame  
//  dst - resultant motion picture  
//  args - optional parameters  
void   update_mhi( IplImage* img, IplImage* dst, int diff_threshold , IplImage* im){
	int countmotion = 0;
    double timestamp = (double)clock()/CLOCKS_PER_SEC; // ��ȡ��ǰʱ��,����Ϊ��λ  
    CvSize size = cvSize(img->width,img->height); // ��ȡ��ǰ֡�ߴ�  
    int i, idx1 = last, idx2;  
    IplImage* silh;  
    CvSeq* seq;  
    CvRect comp_rect;  
    double count;  
    double angle;  
    CvPoint center;  
    double magnitude;  
    CvScalar color;  

    // ��ʼʱΪͼ������ڴ� or ֡�ߴ�ı�ʱ���·����ڴ�  
    if( !mhi || mhi->width != size.width || mhi->height != size.height ){   //����˶���ʷͼ�񲻷���Ҫ�����    
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

    cvCvtColor( img, buf[last], CV_BGR2GRAY ); //RGB֡ͼ���ʽת��Ϊgray  

    idx2 = (last + 1) % N; // index of (last - (N-1))th frame  
    last = idx2;  

    silh = buf[idx2];  
    // ������֡�Ĳ�  
    cvAbsDiff( buf[idx1], buf[idx2], silh );  
    cvThreshold( silh, silh, diff_threshold, 1, CV_THRESH_BINARY ); // �Բ�ͼ������ֵ��  
    cvUpdateMotionHistory( silh, mhi, timestamp, MHI_DURATION ); // �����˶���ʷ  
	/*����ֻ�Ǹ��� ���ص���˶���ʷ��Ҳ��˵���µĲ���ͼ�񣬶��Ƕ�ͼ�������ص��˶�����ĸ��¡�
����silh��x,y�� !=0ʱ���������ص㷢���˶�������Ҫ������и��£���mhi(x,y) = timestamp ��ʾ�˶�������ʱ��
����silh��x,y�� =0ʱ���������ص�δ�����˶�����������Ըõ�ĸ���ʱ���Ƿ񳬹���Ԥ��������ʱ�䣬���ж�mhi(x,y)��timestamp -duration�Ĵ�С����ʱmhi(x,y)��Ϊ�õ����һ�η����˶���ʱ��ֵ������С��timestamp-duration,��ʾ�õ��˶�ʱ���� ����������ʱ�䣬�ʿ���������
��������mhi(x,y)���ڻ��ߵ���timestamp-durationʱ����ʾ�õ�˿���δ�����˶��������ڸ���ʱ���ڣ����Բ��Ըõ㷢���˶���ʱ���ǽ��в�����
	*/
	im = cvCreateImage(cvGetSize(mhi), mhi->depth, mhi->nChannels);
	cvCopy(mhi, im, NULL);
    // convert MHI to blue 8u image  
    // cvCvtScale�ĵ��ĸ����� shift = (MHI_DURATION - timestamp)*255./MHI_DURATION  
    // ����֡�����ʧ����  
    cvCvtScale( mhi, mask, 255./MHI_DURATION, (MHI_DURATION - timestamp)*255./MHI_DURATION );  
    cvZero( dst );  
    cvMerge( mask, 0, 0, 0, dst );

  // B,G,R,0 convert to BLUE image  

    // �����˶����ݶȷ����Լ���ȷ�ķ�������  
    // Filter size = 3  
    cvCalcMotionGradient( mhi, mask, orient,  
                          MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );  
	//�����˶���ʷͼ����ݶȷ��� (maskͼ�񣺱�ע�˶��ݶ�������ȷ�ĵ�)
    if( !storage )  
        storage = cvCreateMemStorage(0);  
    else  
        cvClearMemStorage(storage);  

    // �˶��ָ ����˶���������������  
    seq = cvSegmentMotion( mhi, segmask, storage, timestamp, MAX_TIME_DELTA );  
	countmotion = 0;
    for( i = -1; i < seq->total; i++ ){  
        if( i < 0 ) {        // ������ͼ�����  
            comp_rect = cvRect( 0, 0, size.width, size.height );  
            color = CV_RGB(255,255,255);  
            magnitude = 100;  // ���߳����Լ�Բ�뾶�Ĵ�С����  
        }  
        else {          // ��i���˶����  
            comp_rect = ((CvConnectedComp*)cvGetSeqElem( seq, i ))->rect;  
            // ȥ��С�Ĳ���  
            if( comp_rect.width + comp_rect.height < 100 )  
                continue;  
            color = CV_RGB(255,0,0);  
            magnitude = 30;  
            //if(seq->total > 0) MessageBox(NULL,"Motion Detected",NULL,0);  
        }  
        // ѡ�����ROI  
        cvSetImageROI( silh, comp_rect );   //��ͼ����������Ȥ���� 
        cvSetImageROI( mhi, comp_rect );  
        cvSetImageROI( orient, comp_rect );  
        cvSetImageROI( mask, comp_rect );  

        // ��ѡ��������ڣ������˶�����  
       angle = cvCalcGlobalOrientation( orient, mask, mhi, timestamp,MHI_DURATION);  
	   cout << angle << endl;
       angle = 360.0 - angle;  // adjust for images with top-left origin  
	   cout <<"angle = "<< angle <<"total="<< seq->total<<"i="<<i<< endl;
	   if (angle > 225.0&&angle < 315.0) {
		   countmotion++;
	   }
        // �������ڼ������  
        // Norm(L1) = ��������ֵ�ĺ�  
        count = cvNorm( silh, 0, CV_L1, 0 );  
        cvResetImageROI( mhi );  
        cvResetImageROI( orient );  
        cvResetImageROI( mask );  
        cvResetImageROI( silh );  

        // ���С�˶�������  
        if( count < comp_rect.width*comp_rect.height * 0.05 )  //  ���ص�5%  
            continue;  
        // ��һ������ͷ�ļ�¼�Ա�ʾ����  
		cout << "����Ȥ������comp_recx.x=" << comp_rect.x << "  " << "comp_rect.y" << comp_rect.y << "  " << "comp_rect.width" << comp_rect.width << " " <<"comp_rect.height" << comp_rect.height << "  " <<endl;
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
		cout << "��������ˤ����"<<endl;
		cout << countmotion << endl;
		cin >> countmotion;
	}
}  
#define threshold_diff1 10 //���ü�֡���ֵ
#define threshold_diff2 10 //���ü�֡���ֵ
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
	std::vector<uchar *> mFlag; //���ڱ����Ҫɾ���ĵ�
								//�Ե���
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
				cout << "�˵�(" << i << "," << j << ")" << endl;
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
	int count = 0;  //��¼��������
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //���ƴ������ҵ�����������
			break;
		std::vector<uchar *> mFlag; //���ڱ����Ҫɾ���ĵ�
									//�Ե���
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��
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
						//���
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���
		}

		//�Ե���
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��
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
						//���
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���
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
	int k;  //����ѭ���������ݵ��ⲿ
	for (int i = 0; i < height - 1; i++)
	{
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < width - 1; j++)
		{
			if (type == 0)
			{
				//������
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
				//������
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
				//������
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
				//������
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
	// �������ȸ�ʴ������,��ʴ����������,���Ϳ����޸��ѷ�
	cvMorphologyEx(mask, mask, NULL, NULL, CV_MOP_CLOSE, CVCLOSE_ITR);
	cvMorphologyEx(mask, mask, NULL, NULL, CV_MOP_CLOSE, CVCLOSE_ITR);
	cvMorphologyEx(mask, mask, NULL, NULL, CV_MOP_CLOSE, CVCLOSE_ITR);
	// �����������ͺ�ʴ,֮�����ڿ�����֮��,��Ϊ������ͺ��ٸ�ʴ,�ǲ�����ȥ����
	//FIND CONTOURS AROUND ONLY BIGGER REGIONS
	if (mem_storage == NULL)
		mem_storage = cvCreateMemStorage(0);
	else cvClearMemStorage(mem_storage);
	//cvFindContours(mask,mem_storage,&contours,sizeof(CvContour),CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE,cvPoint(0,0) );
	CvContourScanner scanner = cvStartFindContours(mask, mem_storage, sizeof(CvContour), CV_RETR_EXTERNAL);
	//     ��֮ǰ�����۹�������������,���ֻ��ȡ����û��cvFindContours()���ķ���
	//     ������Ҫ����������ֱ�Ӳ���,�������ַ������ܸ�ǿ��һ��
	CvSeq* c;
	int numCont = 0;
	while ((c = cvFindNextContour(scanner)) != NULL)
	{
		double len = cvContourPerimeter(c);
		// ���������ܳ�
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
			// �滻����ɨ��������ȡ������
			numCont++;
			// ������Ŀ
		}
	}

	contours = cvEndFindContours(&scanner);
	// ����ɨ����̣����ҷ�����߲�ĵ�һ��������ָ��

	// PAINT THE FOUND REGIONS BACK INTO THE IMAGE
	cvZero(mask);
	// ����ģͼ������
	// ��ģ: ָ����0��1��ɵ�һ��������ͼ��

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
		cvAcc(Iscratch, IavgF[number]);//��2��ͼ����ӣ�IavgF[number]=IavgF[number]+Iscratch��IavgF[]����װ����ʱ������ͼƬ���ۼ�
		cvAbsDiff(Iscratch, IprevF[number], Iscratch2);//��2��ͼ�������Iscratch2=abs(Iscratch-IprevF[number]);
		cvAcc(Iscratch2, IdiffF[number]);//IdiffF[]����װ����ͼ�����ۻ���
		Icount[number] += 1.0;//�ۻ���ͼƬ֡������
	}
	first = 0;
	cvCopy(Iscratch, IprevF[number]);//ִ����ú����󣬽���ǰ֡���ݱ���Ϊǰһ֡����
}

void scaleHigh(float scale, int num)//�趨������ģʱ�ĸ���ֵ����
{
	cvConvertScale(IdiffF[num], Iscratch, scale); //Converts with rounding and saturation
	cvAdd(Iscratch, IavgF[num], IhiF[num]);//��ƽ���ۻ�ͼ��������ۻ�ͼ������scale��Ȼ�������
	cvSplit(IhiF[num], Ihi1[num], Ihi2[num], Ihi3[num], 0);//#define cvCvtPixToPlane cvSplit,��cvSplit�ǽ�һ����ͨ������ת��Ϊ������ͨ������
}

void scaleLow(float scale, int num)//�趨������ģʱ�ĵ���ֵ����
{
	cvConvertScale(IdiffF[num], Iscratch, scale); //Converts with rounding and saturation
	cvSub(IavgF[num], Iscratch, IlowF[num]);//��ƽ���ۻ�ͼ��������ۻ�ͼ������scale��Ȼ�������
	cvSplit(IlowF[num], Ilow1[num], Ilow2[num], Ilow3[num], 0);
}

//Once you've learned the background long enough, turn it into a background model
void createModelsfromStats()
{
	for (int i = 0; i<NUM_CAMERAS; i++)
	{
		cvConvertScale(IavgF[i], IavgF[i], (double)(1.0 / Icount[i]));//�˴�Ϊ����ۻ����ͼ���ƽ��ֵ
		cvConvertScale(IdiffF[i], IdiffF[i], (double)(1.0 / Icount[i]));//�˴�Ϊ����ۼ����ͼ���ƽ��ֵ
		cvAddS(IdiffF[i], cvScalar(1.0, 1.0, 1.0), IdiffF[i]);  //Make sure diff is always something��cvAddS������һ����ֵ��һ���������
		scaleHigh(0.7, i);//HIGH_SCALE_NUM��ʼ����Ϊ7����ʵ����һ������
		scaleLow(0.6, i);//LOW_SCALE_NUM��ʼ����Ϊ6
	}
}

void backgroundDiff(IplImage *I, IplImage *Imask, int num)  //Mask should be grayscale
{
	cvCvtScale(I, Iscratch, 1, 0); //To float;
								   //Channel 1
	cvSplit(Iscratch, Igray1, Igray2, Igray3, 0);
	cvInRange(Igray1, Ilow1[num], Ihi1[num], Imask);//Igray1[]����Ӧ�ĵ���Ilow1[]��Ihi1[]֮��ʱ��Imask����Ӧ�ĵ�Ϊ255(��������)
													//Channel 2
	cvInRange(Igray2, Ilow2[num], Ihi2[num], Imaskt);//Ҳ����˵����ÿһ��ͼ��ľ���ֵ��С�ھ���ֵ��ƽ��ֵ��6�����ߴ��ھ���ֵ��ƽ��ֵ��7������Ϊ��ǰ��ͼ��
	cvOr(Imask, Imaskt, Imask);
	//Channel 3
	cvInRange(Igray3, Ilow3[num], Ihi3[num], Imaskt);//����Ĺ̶���ֵ6��7̫�������ˣ����ù��̺�����Ը���ʵ������ֶ�������
	cvOr(Imask, Imaskt, Imask);
	//Finally, invert the results
	cvSubRS(Imask, cvScalar(255), Imask);//ǰ����255��ʾ�ˣ���������0��ʾ
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
	IplImage* pFrImg = NULL; //��ȡ��ǰ��ͼ�񣬼��˶�Ŀ��
	IplImage* pBkImg = NULL; //����ͼ��
	IplImage *ImaskAVG = 0, *ImaskAVGCC = 0;
	IplImage* frame;//����IplImage����ָ�룬���������ڴ�ռ������ÿһ֡ͼ��
	CvMat* pFrameMat = NULL; //ԭʼ��Ƶ����
	CvMat* pFrMat = NULL;    //ǰ������
	CvMat* pBkMat = NULL;    //��������
							 //��������
	cvNamedWindow("video", 1);
	cvNamedWindow("background", 1);
	cvNamedWindow("foreground", 1);
	cvNamedWindow("AVG_ConnectComp", 1);//ƽ������ͨ����������ͼ��
	cvNamedWindow("ForegroundAVG", 1);//ƽ������ͼ��
	maxMod[0] = 3;  //Set color thresholds to default values
	minMod[0] = 10;
	maxMod[1] = 1;
	minMod[1] = 1;
	maxMod[2] = 1;
	minMod[2] = 1;
	float scalehigh = 0.7;//Ĭ��ֵΪ6
	float scalelow = 0.6;//Ĭ��ֵΪ7
	while (1) {
		frame = cvQueryFrame(capture);// ������ͷ��ץȡ������ÿһ֡
		if (!frame) break;//���ץȡ֡Ϊ��   break ����ѭ������ץȡ����һ֡��ʾ�ڴ����Ĵ�����
		char image_name[25];
		sprintf(image_name, "%s%d%s", "E:", ++i, ".jpg");//�����ͼƬ��
		cvSaveImage(image_name, frame);   //����һ֡ͼƬ
		cvWaitKey(3);//��ʱ��ÿ����Լ33֡���������۹ۿ��ٶȣ�
		if (i == 1) {
		}
		else if (i == 2)
		{
			//AVG METHOD ALLOCATION
			AllocateImages(frame);//Ϊ�㷨��ʹ�÷����ڴ�
			scaleHigh(scalehigh);//�趨������ģʱ�ĸ���ֵ����
			scaleLow(scalelow);//�趨������ģʱ�ĵ���ֵ����
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
						accumulateBackground(frame);//ƽ�����ۼӹ���
					}
					if (i == endcapture) {
						createModelsfromStats();//ƽ������ģ����
					}
					if (i >= endcapture) {//endcapture֡��ʼ���ǰ��
						backgroundDiff(frame, ImaskAVG);
						sprintf(image_name, "%s%d%s", "/Users/jiaoyukun/Downloads/opencv/", i, "(2).jpg");//�����ͼƬ��
						cvSaveImage(image_name, ImaskAVGCC);   //����һ֡ͼƬ
						cvCopy(ImaskAVG, ImaskAVGCC);
						cvconnectedComponents(ImaskAVGCC, 1, 6.0);//ƽ�����е�ǰ�����
						sprintf(image_name, "%s%d%s", "/Users/jiaoyukun/Downloads/opencv/", i, "(3).jpg");//�����ͼƬ��
						cvSaveImage(image_name, ImaskAVGCC);   //����һ֡ͼƬ
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
						sprintf(image_name, "%s%d%s", "/Users/jiaoyukun/Downloads/opencv/", i, "(4).jpg");//�����ͼƬ��
						cvSaveImage(image_name, &img);   //����һ֡ͼƬ
						cv::Mat src2 = cv::cvarrToMat(&img);
						cv::threshold(src2, src2, 128, 1, cv::THRESH_BINARY);
						cv::Mat dst = thinImage(src2);//��ʾͼ��
						dian2(dst);
						dst = dst * 255;
						IplImage img2 = IplImage(dst);
						sprintf(image_name, "%s%d%s", "/Users/jiaoyukun/Downloads/opencv/", i, "(5).jpg");//�����ͼƬ��
						cvSaveImage(image_name, &img2);   //����һ֡ͼƬ
														  //cvWaitKey(2);
					}
				}
			}
		}
	}
	//���ٴ���
	cvDestroyWindow("video");
	cvDestroyWindow("background");
	cvDestroyWindow("foreground");
	cvDestroyWindow("Example2");
	cvDestroyWindow("ForegroundAVG");
	cvDestroyWindow("AVG_ConnectComp");
	cvDestroyWindow("ForegroundCodeBook");
	cvDestroyWindow("CodeBook_ConnectComp");
	//�ͷ�ͼ��;���
	cvReleaseImage(&ImaskAVG);
	cvReleaseImage(&ImaskAVGCC);
	cvReleaseImage(&pFrImg);
	cvReleaseImage(&pBkImg);
	cvReleaseMat(&pFrameMat);
	cvReleaseMat(&pFrMat);
	cvReleaseMat(&pBkMat);
	DeallocateImages();//�ͷ�ƽ����������ģ�������õ����ڴ�
	cvReleaseCapture(&capture);//�ͷ��ڴ棻
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