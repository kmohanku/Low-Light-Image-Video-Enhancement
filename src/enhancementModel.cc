#include "../include/enhancementModel.h"
#include <iostream>

EnhancementModel::EnhancementModel(const std::string& path, const std::string& ip_node = "serving_default_input:0", 
    const std::string& out_node = "StatefulPartitionedCall:1") : 
    model_path(path), input_node(ip_node), output_node(out_node){               
    session_options.config.mutable_gpu_options()->set_allow_growth(true);
    tensorflow::Status status = LoadSavedModel(session_options, run_options, model_path, {tensorflow::kSavedModelTagServe}, &model_bundle);
    if (!status.ok()){
        std::cerr << "Failed to load model! Check path. " << status;
    }
    else {
        std::cout << "Model Loaded Succesfully!\n";
    }
}

tensorflow::Tensor EnhancementModel::convertMatToTensor(const cv::Mat& frame){
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, frame.rows, frame.cols, frame.channels()}));
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();
    const float * pixel_values = (float*) frame.data;
    // change float pointers. No copy
    for(int dim1 = 0; dim1 < frame.rows; ++dim1){
        const float* pixel_row = pixel_values + (dim1 * frame.cols * frame.channels());
        for(int dim2 = 0; dim2 < frame.cols; ++dim2){
            const float* pixel_cols = pixel_row + (dim2 * frame.channels());
            for(int channel = 0; channel < frame.channels(); ++channel){
                const float* channel_value = pixel_cols + channel;
                input_tensor_mapped(0, dim1, dim2, channel) = *channel_value;
            }
        }
    }
    return input_tensor;
}

void EnhancementModel::enhanceImage(std::vector<tensorflow::Tensor>& input, std::vector<tensorflow::Tensor>* output){
    std::vector<std::string> output_node_vec = {output_node};
    tensorflow::Status runStatus = model_bundle.GetSession()->Run({{input_node, input[0]}}, output_node_vec, {}, output);
    if(!runStatus.ok()){
        std::cout << "Evaluation Failed. Skipping frame....\n";
    }
}

cv::Mat EnhancementModel::runModel(const cv::Mat& frame){
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    cv::Mat float_img;
    frame.convertTo(float_img, CV_32F);
    float_img *= (1 / max_pixel);
    tensorflow::Tensor input_image = convertMatToTensor(float_img);
    std::vector<tensorflow::Tensor> output_tensor;
    std::vector<tensorflow::Tensor> input_tensor{input_image};
    enhanceImage(input_tensor, &output_tensor);
    cv::Mat output_image(output_tensor[0].dim_size(1), output_tensor[0].dim_size(2), CV_32FC3, output_tensor[0].flat<float>().data());
    output_image *= max_pixel;
    cv::Mat enhanced_image;
    output_image.convertTo(enhanced_image, CV_8UC3);
    cv::cvtColor(enhanced_image, enhanced_image, cv::COLOR_RGB2BGR);
    return enhanced_image;
}
