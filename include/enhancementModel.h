#ifndef _ENHANCEMENT_MODEL_H_
#define _ENHANCEMENT_MODEL_H_

#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

class EnhancementModel{
	private:

		const std::string model_path;
    	const std::string input_node;
    	const std::string output_node;
    	const float max_pixel = 255.0;

		tensorflow::SavedModelBundleLite model_bundle;
    	tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
    	tensorflow::RunOptions run_options = tensorflow::RunOptions();

    	void enhanceImage(std::vector<tensorflow::Tensor>&, std::vector<tensorflow::Tensor>*);

    public:

    	EnhancementModel(const std::string&, const std::string&, const std::string&);
    	tensorflow::Tensor convertMatToTensor(const cv::Mat&);
    	cv::Mat runModel(const cv::Mat&);
};

#endif