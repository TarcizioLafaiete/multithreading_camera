#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <memory>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <queue>

struct cameraManager{
    std::unique_ptr<cv::VideoCapture> camera;
    std::atomic<bool> stop_thread{false};
};

struct ringBuffer{
    std::unique_ptr<std::queue<cv::Mat>> imageQueue;
    std::atomic<int> current_consumers;
    int total_consumers;
    int maxlen;
};

cameraManager manager;
ringBuffer imageRingBuffer;


std::mutex imageQueue_mutex;
std::mutex manager_mutex;
std::condition_variable queue_cond;

void captureVideo(std::string videoIndex){

    manager.camera.reset(new cv::VideoCapture());

    manager.camera->open(videoIndex);
    manager.camera->set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('Y','U','Y','V'));

    while(!manager.stop_thread){
        cv::Mat frame;
        bool ret;
        {
            std::unique_lock<std::mutex> lock(manager_mutex);
            ret = manager.camera->read(frame);        
        }

        if(ret){
            {
                std::unique_lock<std::mutex> lock(imageQueue_mutex);
                if(imageRingBuffer.imageQueue->size() == imageRingBuffer.maxlen){
                    imageRingBuffer.imageQueue->pop();
                }
                imageRingBuffer.imageQueue->push(frame);
                queue_cond.notify_one();
            }
        }
    }

}

void viewImage(){

    cv::Mat frame;
    while(true){
        {
            std::unique_lock<std::mutex> lock(imageQueue_mutex);
            queue_cond.wait(lock,[] {return !imageRingBuffer.imageQueue->empty();});

            frame = imageRingBuffer.imageQueue->front();
            imageRingBuffer.imageQueue->pop();
        }

        cv::imshow("display",frame);
        cv::waitKey(1);
    }

}

int main(int argc, char* argv[]){

    std::string videoIndex = argv[1];

    imageRingBuffer.imageQueue.reset(new std::queue<cv::Mat>());
    imageRingBuffer.maxlen = 10;
    imageRingBuffer.total_consumers = 2;

    std::thread captureThread(captureVideo,videoIndex);
    std::thread viewThread(viewImage);

    captureThread.join();
    viewThread.join();


}