#include "../include/audioHandler.h"
#include <iostream>
#include <string>
AudioHandler::AudioHandler(const std::string& src, const std::string& dst):
	source_path(src), destination_path(dst){
	audio_strip[2] = source_path;
	audio_attach.back() = destination_path;
}

void AudioHandler::stripAudio(){
	std::string cmd = "";
	for(auto& st : audio_strip){
		cmd+=st + ' ';
	}
	cmd.pop_back();
	system(cmd.c_str());
	return;
}

void AudioHandler::addAudio(){
	std::string cmd = "";
	for(auto& st : audio_attach){
		cmd+=st + ' ';
	}
	cmd.pop_back();
	system(cmd.c_str());
	remove("temp.mp4");
	remove("output-audio.aac");
	return;
}