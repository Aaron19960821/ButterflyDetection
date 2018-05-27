/*************************************************************************
    > File Name: yolo.h
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Fri May 18 21:06:42 2018
 ************************************************************************/

#include<iostream>
#include<cstdio>
#include<cstring>
#include<string>
#include<algorithm>

class yolo
{
	public:
		yolo(char* rootDir);
		void train(char* target);
		void predict(char* target, char* imageList);
	private:
		void writeBox(image im, detection* dets, int num, float thresh, char** names, int classes);
		int status;
		std::string datacfg;
		std::string cfgfile;
		std::string weightfile;
		std::string pretrainedModel;
		int* gpu;
		int ngpus;
};

