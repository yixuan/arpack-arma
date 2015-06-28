CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
CPPFLAGS = -I../include
LDFLAGS =
LIBS = -llapack -lblas

HEADERS = $(wildcard ../include/*.h)

.PHONY: all test clean

all: QR.out LDL.out LU.out SymEigs.out SymEigsShift.out GenEigs.out

test:
	./QR.out
	./LDL.out
	./LU.out
	./SymEigs.out
	./SymEigsShift.out
	./GenEigs.out

%.out: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm *.out
