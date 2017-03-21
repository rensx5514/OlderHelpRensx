#pragma once
//
//  wj.hpp
//  opencv
//
//  Created by ��ع�� on 16/9/7.
//  Copyright  2016�� ��ع��. All rights reserved.
//

#ifndef wj_hpp
#define wj_hpp

#include <iostream>
#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>



//IMPORTANT DEFINES:
#define NUM_CAMERAS   1              //This function can handle an array of cameras
#define HIGH_SCALE_NUM 1.0            //How many average differences from average image on the high side == background
#define LOW_SCALE_NUM 1.0        //How many average differences from average image on the low side == background

void AllocateImages(IplImage *I);
void DeallocateImages();
void accumulateBackground(IplImage *I, int number = 0);
void scaleHigh(float scale = HIGH_SCALE_NUM, int num = 0);
void scaleLow(float scale = LOW_SCALE_NUM, int num = 0);
void createModelsfromStats();
void backgroundDiff(IplImage *I, IplImage *Imask, int num = 0);

#endif /* wj_hpp */
/*
//������ͷ��֡��ȡͼƬ
cvNamedWindow("Example2", CV_WINDOW_AUTOSIZE);//�������ڣ������֣�Ĭ�ϴ�С��
CvCapture *capture = NULL;// ��Ƶ��ȡ�ṹ��������Ϊ��Ƶ��ȡ������һ������
capture = cvCreateFileCapture("E:\\916.mp4");//������ͷ��������ͷ�л�ȡ��Ƶ
//֡��
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
if (!frame) {
int t = 0;
for (t = 0; t < 100; t++) {
printf("111111111111111111111");
}
break;
}//���ץȡ֡Ϊ��   break ����ѭ������ץȡ����һ֡��ʾ�ڴ����Ĵ�����
char image_name[25];
sprintf(image_name, "%s%d%s", "E:/oimage/asd", ++i, ".jpg");//�����ͼƬ��
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
cvCopy(ImaskAVG, ImaskAVGCC);
cvconnectedComponents(ImaskAVGCC, 1, 6.0);//ƽ�����е�ǰ�����
//hh(ImaskAVGCC);
//cvThin(ImaskAVGCC,ImaskAVGCC,88);
}
}
//cvShowImage( "AVG_ConnectComp",ImaskAVGCC);
//cvShowImage( "ForegroundAVG",ImaskAVG);
sprintf(image_name, "%s%d%s", "E:/oimage/asd", i, "(2).jpg");//�����ͼƬ��
cvSaveImage(image_name, ImaskAVGCC);   //����һ֡ͼƬ
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

*/#pragma once
