HEADERS = $(wildcard ../include/*.h)

all: QR.out LDL.out SymEigs.out SymEigsShift.out

QR.out: $(HEADERS) QR.cpp
	g++ -O2 -I../include QR.cpp -llapack -lblas -o QR.out

LDL.out: $(HEADERS) LDL.cpp
	g++ -O2 -I../include LDL.cpp -llapack -lblas -o LDL.out

SymEigs.out: $(HEADERS) SymEigs.cpp
	g++ -O2 -I../include SymEigs.cpp -llapack -lblas -o SymEigs.out

SymEigsShift.out: $(HEADERS) SymEigsShift.cpp
	g++ -O2 -I../include SymEigsShift.cpp -llapack -lblas -o SymEigsShift.out

clean:
	rm *.out
