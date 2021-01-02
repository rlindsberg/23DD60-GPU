# instructions
kinit --forwardable ruliu@NADA.KTH.SE
salloc --nodes=1 -t 00:55:00 -A edu20.DD2360

module load cuda

nvcc -O3 -arch=sm_30 kernel.cu -o akernel && srun -n 1 ./akernel images/hw.bmp && display -resize 1280x720 images/hw3_result_1.bmp
display -resize 1280x720 images/hw3_result_1.bmp
