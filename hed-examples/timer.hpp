#include <sys/time.h>
#include <iostream>

class Timer {

private:
    timeval startTime;

public:
    void start() {
        gettimeofday(&startTime, NULL);
    }

    double stop() {
        timeval endTime;
        long seconds, useconds;
        double duration;

        gettimeofday(&endTime, NULL);

        seconds = endTime.tv_sec - startTime.tv_sec;
        useconds = endTime.tv_usec - startTime.tv_usec;

        duration = seconds + useconds / 1000000.0;

        std::cout << "Elapsed " << duration <<  " seconds" << std::endl;

        return duration;
    }
};
