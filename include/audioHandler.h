#ifndef _AUDIO_HANDLER_H_
#define _AUDIO_HANDLER_H_

#include <cstdlib>
#include <string>
#include <vector>

class AudioHandler{
	private:
		std::string source_path;
		std::string destination_path;
		std::vector<std::string> audio_strip = {"ffmpeg", "-i", "", "-vn -acodec copy output-audio.aac"};		
		std::vector<std::string> audio_attach = {"ffmpeg", "-i", "temp.mp4", "-i", "output-audio.aac", "-shortest -c:v copy -c:a aac -b:a 256k", ""};
	public:
		AudioHandler(const std::string&, const std::string&);
		void stripAudio();
		void addAudio();
};

#endif