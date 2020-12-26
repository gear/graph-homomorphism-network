all: test/test.o

test/test.o: test/test.cc src/hom.hh src/graph.hh src/treedec.hh
	g++ -O3 -std=c++17 -I./src -o test/test.o test/test.cc 

test/speedtest.o: test/speedtest.cc src/hom.hh src/graph.hh src/treedec.hh
	g++ -O3 -std=c++17 -I./src -o test/speedtest.o test/speedtest.cc 


.PHONY: test
test: test/test.o test/speedtest.o
	./test/test.o
	./test/speedtest.o
