HEADERS = $(wildcard ../include/*.h)

all: QR SymEigs

QR: $(HEADERS) QR.cpp
	g++ -O2 -I../include QR.cpp -llapack -lblas -o QR

SymEigs: $(HEADERS) SymEigs.cpp
	g++ -O2 -I../include SymEigs.cpp -llapack -lblas -o SymEigs

clean:
	rm QR SymEigs *.o
