CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
CPPFLAGS = -I../../include
LDFLAGS =
LIBS = -llapack -lblas

HEADERS = $(wildcard ../*.hpp)

.PHONY: all test clean

all: QR.out Eigen.out

test:
	-./QR.out
	-./Eigen.out

%.out: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	-rm *.out
