CC = g++
CCFLAGS = -Wall -Wextra -Iinclude -I/opt/cuda/include
NV = nvcc
NVFLAGS = -Iinclude -I/opt/cuda/include

SRC_DIR = src
OBJ_DIR = obj
LSF_DIR = lsf
ARCHIVE_DIR = ../archive

SRC_C = $(wildcard $(SRC_DIR)/*.cpp)
SRC_CU = $(wildcard $(SRC_DIR)/*.cu)
OBJ_C = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_C))
OBJ_CU = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC_CU))

CUDA_LIB_DIR = /opt/cuda/lib
CUDA_LIB = -lcudart -lcuda

LSF = $(wildcard $(LSF_DIR)/*.lsf)




all: $(TARGET)

serial: CCFLAGS += -DSERIAL
serial: NVFLAGS += -DSERIAL
serial: $(OBJ_C)
	$(CC) $(CCFLAGS) -o $@ $^	

parallel: $(OBJ_C) $(OBJ_CU)
	$(NV) $(NVFLAGS) -o $@ $^ -L$(CUDA_LIB_DIR) $(CUDA_LIB)

DEP := $(patsubst $(OBJ_DIR)/%.o, $(OBJ_DIR)/%.d, $(OBJ_C)) \
			 $(patsubst $(OBJ_DIR)/%.o, $(OBJ_DIR)/%.d, $(OBJ_CU))
-include $(DEP)
DEPFLAGS = -MMD -MF $(@:.o=.d)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(OBJ_DIR)
	$(CC) $(CCFLAGS) -c $< -o $@ $(DEPFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	mkdir -p $(OBJ_DIR)
	$(NV) $(NVFLAGS) -c $< -o $@ $(DEPFLAGS)



clean:
	rm -f $(OBJ_C) $(OBJ_CU) $(DEP)

fullclean:
	rm -rf $(OBJ_DIR) $(LSF_DIR) logs err serial parallel



bsubload: $(TARGET)
	@if command -v bsub >/dev/null 2>&1; then \
		JOB_IDS=$$( \
			for lsf_script in $(LSF); do \
				bsub < $$lsf_script | awk '{print $$2}' | sed 's/[<>]//g'; \
			done \
		); \
		echo "$$JOB_IDS"; \
		for job_id in $$JOB_IDS; do \
			bwait -w "done($$job_id)"; \
		done; \
	else \
		for lsf_script in $(LSF); do \
			sh "$$lsf_script"; \
		done; \
	fi

lsf:
	echo $(KBLOCKS) $(KTHREADS) $(EXEC) $(ARRAY_SIZE) $(CYCLES)
	mkdir -p lsf

	$(eval JOB_NAME := "lab4_3_$(EXEC)_$(KBLOCKS)_$(KTHREADS)")
	$(eval PROJECT_NAME := "mothm_lab4_3")
	$(eval LOG_FILE := "$(JOB_NAME).log")

	echo -e "#!/bin/bash\nmkdir -p logs err\n\n#BSUB -J $(JOB_NAME)\n#BSUB -P $(PROJECT_NAME)\n#BSUB -W 08:00\n#BSUB -n 1\n#BSUB -oo logs/$(LOG_FILE)\n#BSUB -eo err/err_$(LOG_FILE)\n\nexport ARRAY_SIZE=$(ARRAY_SIZE)\nexport CYCLES=$(CYCLES)\nexport KBLOCKS=$(KBLOCKS)\nexport KTHREADS=$(KTHREADS)\n\nmodule load cuda/11.4\n{ time ./$(EXEC) > logs/$(JOB_NAME).tlog ; } 2>> logs/$(JOB_NAME).tlog" > "./lsf/pr$(KBLOCKS)_$(KTHREADS)_$(EXEC).lsf"

	chmod +x ./lsf/pr$(KBLOCKS)_$(KTHREADS)_$(EXEC).lsf

auto:
	$(eval AR_SIZE := 60000000)
	$(eval CYCLES := 250)
	$(eval THREADS := 16 32 64 128 256 512 1024)
	mkdir -p $(ARCHIVE_DIR)

	make fullclean
	make serial
	make clean
	make parallel

	make lsf EXEC=serial ARRAY_SIZE=$(AR_SIZE) CYCLES=$(CYCLES) KBLOCKS=1 KTHREADS=1
	for thread in $(THREADS) ; do \
		make lsf EXEC=parallel ARRAY_SIZE=$(AR_SIZE) CYCLES=$(CYCLES) KBLOCKS=0 KTHREADS=$$thread ; \
	done
	
	make bsubload
	mv logs/* $(ARCHIVE_DIR)
	mv err/* $(ARCHIVE_DIR)

.PHONY: lsf
