CXX = g++
CXXFLAGS = -Wall -O2
CPPFLAGS = -I..
LDFLAGS =
LIBS = -llapack -lblas

HEADERS = $(wildcard ../armadillo_bits/*.hpp)

.PHONY: all test clean

all: QR.out Eigen.out SymEigs.out GenEigs.out ArmaSymEigs.out ArmaGenEigs.out

test:
	-./QR.out
	-./Eigen.out
	-./SymEigs.out
	-./GenEigs.out
	-./ArmaSymEigs.out
	-./ArmaGenEigs.out

%.out: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	-rm *.out
