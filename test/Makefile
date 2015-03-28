all: QR

QR: QR.cpp
	g++ -O2 -I../include QR.cpp -llapack -lblas -o QR

clean:
	rm QR *.o
