#include <iostream>
#include "../include/enhancementModel.h"
#include "../include/audioHandler.h"
#include <chrono>

int main(int argc, char* argv[])
{   
    if(argc < 4){
        std::cout << "USAGE:\n";
        std::cout << "      ./video-enhancement <model path> <source video> <destination video>\n";
        return -1;
    }

    const std::string model_path = argv[1];
    const std::string source_video = argv[2];
    const std::string dst_video = argv[3];
    const std::string inp = "serving_default_input:0";
    const std::string out = "StatefulPartitionedCall:1";

    EnhancementModel* model = new EnhancementModel(model_path, inp, out);
    AudioHandler* audiotool = new AudioHandler(source_video, dst_video);
    audiotool->stripAudio();
    cv::VideoCapture cap(source_video);
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer;
    int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); 
    if (!cap.isOpened()){
        std::cout << "Cannot open the video file. \n";
        return -1;
    }
    bool firstframe = true;
    int counter = 0;
    auto start = std::chrono::steady_clock::now();
    while(true) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            break;
        }
        if(firstframe){
            writer.open("temp.mp4", codec, fps, cv::Size(frame.cols, frame.rows));
            firstframe = false;
        }
        cv::Mat en_im = model->runModel(frame);
        writer.write(en_im);
    } 
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    cap.release();
    writer.release();
    audiotool->addAudio();
    delete model;
    delete audiotool;
    return 0;
}