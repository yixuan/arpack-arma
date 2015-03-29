all: QR SymEigs

QR: QR.cpp
	g++ -O2 -I../include QR.cpp -llapack -lblas -o QR

SymEigs: SymEigs.cpp
	g++ -O2 -I../include SymEigs.cpp -llapack -lblas -o SymEigs

clean:
	rm QR SymEigs *.o
