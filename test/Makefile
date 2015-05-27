CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
CPPFLAGS = -I../include
LDFLAGS =
LIBS = -llapack -lblas

HEADERS = $(wildcard ../include/*.h)

.PHONY: all test clean

all: QR.out LDL.out SymEigs.out SymEigsShift.out

test:
	./QR.out
	./LDL.out
	./SymEigs.out
	./SymEigsShift.out

%.out: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm *.out
