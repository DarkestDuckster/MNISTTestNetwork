
CC = nvcc
LIBS = -lcublas -lcudnn

HEADERS = fileloader.hu

ODIR = objs
_OBJS = cudamethods.o SMem.o fileloader.o convmethods.o
OBJS = $(patsubst %, $(ODIR)/%, $(_OBJS))

$(ODIR)/%.o: %.cu $(HEADERS)
	$(CC) -c -o $@ $< $(LIBS)

SMem.exe: $(OBJS)
	$(CC) -o $@ $^ $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o	
