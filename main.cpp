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
    std::unique_ptr<cv::VideoWriter> writer;
    std::atomic<bool> stop_thread{false};
};

struct ringBuffer{
    std::unique_ptr<std::queue<cv::Mat>> imageQueue;
    int maxlen;
    std::atomic<bool> current_consumers;
    int total_consumers;
    std::chrono::milliseconds timeout;
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
                queue_cond.wait_for(lock,imageRingBuffer.timeout,[]{return imageRingBuffer.current_consumers.load() == 0;});
                
                if(imageRingBuffer.imageQueue->size() == imageRingBuffer.maxlen){
                    imageRingBuffer.current_consumers.store(imageRingBuffer.total_consumers);
                    imageRingBuffer.imageQueue->pop();
                }
                imageRingBuffer.imageQueue->push(frame);
                queue_cond.notify_all();
            }
        }
    }

    manager.camera->release();

}

void recordImage(){

    cv::Mat frame;

    {
        std::unique_lock<std::mutex> lock(imageQueue_mutex);
        queue_cond.wait(lock,[]{return !imageRingBuffer.imageQueue->empty();});
    }

    int frame_width = static_cast<int>(manager.camera->get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_heigth = static_cast<int>(manager.camera->get(cv::CAP_PROP_FRAME_HEIGHT));

    manager.writer.reset(new cv::VideoWriter());

    manager.writer->open("saida.avi",cv::VideoWriter::fourcc('X','V','I','D'),10,cv::Size(frame_width,frame_heigth));

    while(!manager.stop_thread){
        {
            std::unique_lock<std::mutex> lock(imageQueue_mutex);
            queue_cond.wait(lock,[]{return !imageRingBuffer.imageQueue->empty();});

            frame = imageRingBuffer.imageQueue->front();
            // int x = imageRingBuffer.current_consumers.load();
            imageRingBuffer.current_consumers.store(0);

            queue_cond.notify_all();
        }

            manager.writer->write(frame);
    }


    manager.writer->release();

}

void viewImage(){

    cv::Mat frame;
    while(!manager.stop_thread){
        {
            std::unique_lock<std::mutex> lock(imageQueue_mutex);
            queue_cond.wait(lock,[] {return !imageRingBuffer.imageQueue->empty();});

            frame = imageRingBuffer.imageQueue->front();
            imageRingBuffer.imageQueue->pop();
            // int x = imageRingBuffer.current_consumers.load();
            imageRingBuffer.current_consumers.store(1);

            queue_cond.notify_all();
        }

        cv::imshow("display",frame);
        cv::waitKey(1);
    }

    cv::destroyAllWindows();

}

int main(int argc, char* argv[]){

    std::string videoIndex = argv[1];

    imageRingBuffer.imageQueue.reset(new std::queue<cv::Mat>());
    imageRingBuffer.maxlen = 10;
    imageRingBuffer.current_consumers = 2;
    imageRingBuffer.total_consumers = 2;
    imageRingBuffer.timeout = std::chrono::milliseconds(50);

    std::thread captureThread(captureVideo,videoIndex);
    std::thread viewThread(viewImage);
    std::thread recordThread(recordImage);

    std::string op;
    std::cin>>op;
    std::cout<<"Acabou"<<std::endl;

    manager.stop_thread.store(true);

    if(captureThread.joinable()){
        captureThread.join();
    }

    if(viewThread.joinable()){
        viewThread.join();
    }

    if(recordThread.joinable()){
        recordThread.join();
    }


}