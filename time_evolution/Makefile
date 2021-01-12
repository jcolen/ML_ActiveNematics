m.PHONY : sharpen.o libsharpen.so overflow.o liboverflow.so winding.o libwinding.so

libsharpen.so : sharpen.o
	gcc -shared -Wl,-soname,libsharpen.so -o libsharpen.so sharpen.o

sharpen.o : sharpen.c
	gcc -c -fPIC sharpen.c -o sharpen.o

liboverflow.so : overflow.o
	gcc -shared -Wl,-soname,liboverflow.so -o liboverflow.so overflow.o

overflow.o : sharpen_overflow.c
	gcc -c -fPIC sharpen_overflow.c -o overflow.o

libwinding.so : winding.o
	gcc -shared -Wl,-soname,libwinding.so -o libwinding.so winding.o

winding.o : winding.c
	gcc -c -fPIC winding.c -o winding.o

clean :
	-rm -vf libsharpen.so sharpen.o sharpen.pyc winding.o winding.so overflow.o overflow.so
