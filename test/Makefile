HEADERS = $(wildcard ../include/*.h)

all: QR LDL SymEigs

QR: $(HEADERS) QR.cpp
	g++ -O2 -I../include QR.cpp -llapack -lblas -o QR

LDL: $(HEADERS) LDL.cpp
	g++ -O2 -I../include LDL.cpp -llapack -lblas -o LDL

SymEigs: $(HEADERS) SymEigs.cpp
	g++ -O2 -I../include SymEigs.cpp -llapack -lblas -o SymEigs

clean:
	rm QR LDL SymEigs
