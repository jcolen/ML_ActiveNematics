m.PHONY : sharpen.o libsharpen.so winding.o libwinding.so

libsharpen.so : sharpen.o
	gcc -shared -Wl,-soname,libsharpen.so -o libsharpen.so sharpen.o

sharpen.o : sharpen.c
	gcc -c -fPIC sharpen.c -o sharpen.o

libwinding.so : winding.o
	gcc -shared -Wl,-soname,libwinding.so -o libwinding.so winding.o

winding.o : winding.c
	gcc -c -fPIC winding.c -o winding.o

clean :
	-rm -vf libsharpen.so sharpen.o sharpen.pyc winding.o winding.so
