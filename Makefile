all: a3

CXXFLAGS+=-g -I ./ -lX11 -lopencv_core -lpthread

a3: CImg.h a3.cpp Classifier.h NearestNeighbor.h BagofWords.h Kmeans.h
	g++ $(OPT) a3.cpp -o a3 -lX11 -lopencv_core -lpthread -I. -Isiftpp siftpp/sift.cpp

clean:
	rm a3
