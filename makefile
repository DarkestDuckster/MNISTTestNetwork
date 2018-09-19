
CC = nvcc
LIBS = -lcublas -lcudnn
INCLUDES = -I.
APPENDS = $(LIBS) $(INCLUDES)

EXE = mnist.exe

COMP_STEP = $(CC) -dc -o $@ $< $(APPENDS)

ODIR = objs
_OBJS = cudamethods.o SMem.o fileloader.o
OBJS = $(patsubst %, $(ODIR)/%, $(_OBJS))

CUBLASDIR = cublas
CUBLASOBJDIR = $(CUBLASDIR)/$(ODIR)
_CUBLASOBJS = cublasmethods.o dense.o transpose.o
CUBLASOBJS = $(patsubst %, $(CUBLASOBJDIR)/%, $(_CUBLASOBJS))

CONVDIR = convolutional
CONVOBJDIR = $(CONVDIR)/$(ODIR)
_CONVOBJS = convmethods.o crossentropy.o
CONVOBJS = $(patsubst %, $(CONVOBJDIR)/%, $(_CONVOBJS))

OBJDIRS = $(ODIR) $(CUBLASOBJDIR) $(CONVOBJDIR)

$(ODIR)/%.o: %.cu
	$(COMP_STEP)

$(CUBLASOBJDIR)/%.o: $(CUBLASDIR)/%.cu
	$(COMP_STEP)

$(CONVOBJDIR)/%.o: $(CONVDIR)/%.cu
	$(COMP_STEP)

.PHONY: all
all: $(EXE)

.PHONY: run
run: all
	./$(EXE)

$(EXE): $(CONVOBJS) $(CUBLASOBJS) $(OBJS)
	$(CC) -o $@ $^ $(APPENDS)

.PHONY: clean
clean:
	rm -f $(EXE)
	$(foreach dir,$(OBJDIRS),rm -f $(dir)/*.o)
