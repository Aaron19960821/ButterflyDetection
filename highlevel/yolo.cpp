/*************************************************************************
    > File Name: yolo.cpp
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Fri May 18 21:12:18 2018
 ************************************************************************/

#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<fstream>
#include<string>
#include<assert.h>

#include"darknet.h"
#include"yolo.h"
#include"../json/json/json.h"


yolo::yolo(char* rootDir)
{
	Json::Value root;
	Json::Reader reader;

	std::string jsonPath = rootDir;
	if(jsonPath[jsonPath.length()-1] != '/')
	{
		jsonPath += '/';
	}
	
	std::ifstream ifs(jsonPath + "model.json");
	if(!ifs.is_open())
	{
		status = 1;
		ifs = ifstream(jsonPath + "build.json");
	}
	else
	{
		status = 2;
	}
	assert(ifs.is_open());

	if(!reader.parse(ifs, root))
	{
		throw "Can not parse json file..";
	}
	else
	{
		datacfg = jsonPath + root["datacfg"].asString();
		cfgfile = jsonPath + root["cfgfile"].asString();
		if(status == 1)
		{
			if(root.isMember("pretrainedmodel"))
			{
				pretrainedModel = jsonPath + std::string(root["pretrainedmodel"].asString());
			}
			else
			{
				pretrainedModel = NULL;
			}
		}
		else if(status == 2)
		{
			weightfile = jsonPath + std::string(root["weightfile"].asString());
		}
	}
}

void yolo::train(char* target)
{
	//if and only if status equals 1 
	//can we start train yolo
	assert(status == 1);
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = target;

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);
        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void yolo::predict(char* target, char* imgList)
{
		list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.45;

		ifstream ifs;
		ifs.open(imgList, std::ifstream::in);

		assert(ifs.is_open());

    while(ifs >> input)
		{
			std::cout << input << std::endl;
			image im = load_image_color(input,0,0);
			image sized = letterbox_image(im, net->w, net->h);
			layer l = net->layers[net->n-1];

			float *X = sized.data;
			time=what_time_is_it_now();
			network_predict(net, X);
			printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
			int nboxes = 0;
			detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
			if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
			free_detections(dets, nboxes);
			free_image(im);
			free_image(sized);
    }
		ifs.close();
}

void writeBox(image im, detection* dets, int num, float thresh, char** names, int classes)
{
		int i,j;
    for(i = 0; i < num; ++i)
		{
        char labelstr[4096] = {0};
        int class = -1;
        for(j = 0; j < classes; ++j)
				{
            if (dets[i].prob[j] > thresh)
						{
                if (class < 0) 
								{
                    strcat(labelstr, names[j]);
                    class = j;
                } else 
								{
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f ", names[j], dets[i].prob[j]*100);
            }
        }
        if(class >= 0){
            int width = im.h * .006;

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;
						printf("%d %d %d %d\n", left, right, top, bot);
        }
    }
}

