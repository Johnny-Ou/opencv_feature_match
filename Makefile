prefix=/usr/local/Cellar/opencv/4.5.3
includedir=${prefix}/include/opencv4

CFLAGS = -std=c++11 `pkg-config --cflags opencv4` 
LIBS = `pkg-config --libs opencv4`



PROG = feature_match
all: $(PROG)
% : %.cpp
	g++ $(CFLAGS) $(LIBS) -o $@ $< 