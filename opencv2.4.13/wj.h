#pragma once
//
//  wj.hpp
//  opencv
//
//  Created by 焦毓堃 on 16/9/7.
//  Copyright  2016年 焦毓堃. All rights reserved.
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
//打开摄像头分帧获取图片
cvNamedWindow("Example2", CV_WINDOW_AUTOSIZE);//创建窗口，（名字，默认大小）
CvCapture *capture = NULL;// 视频获取结构，用来作为视频获取函数的一个参数
capture = cvCreateFileCapture("E:\\916.mp4");//打开摄像头，从摄像头中获取视频
//帧数
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
if (!frame) {
int t = 0;
for (t = 0; t < 100; t++) {
printf("111111111111111111111");
}
break;
}//如果抓取帧为空   break 打破循环否则将抓取的那一帧显示在创建的窗口上
char image_name[25];
sprintf(image_name, "%s%d%s", "E:/oimage/asd", ++i, ".jpg");//保存的图片名
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
cvCopy(ImaskAVG, ImaskAVGCC);
cvconnectedComponents(ImaskAVGCC, 1, 6.0);//平均法中的前景清除
//hh(ImaskAVGCC);
//cvThin(ImaskAVGCC,ImaskAVGCC,88);
}
}
//cvShowImage( "AVG_ConnectComp",ImaskAVGCC);
//cvShowImage( "ForegroundAVG",ImaskAVG);
sprintf(image_name, "%s%d%s", "E:/oimage/asd", i, "(2).jpg");//保存的图片名
cvSaveImage(image_name, ImaskAVGCC);   //保存一帧图片
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

*/#pragma once
