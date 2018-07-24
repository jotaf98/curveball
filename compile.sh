### Compile pool-switches as it is wrapped in matlab
### Please specify the Matlab path or the default one
### /usr/local/MATLAB/R2017b will be used

# specify a different Matlab path by passing it as an argument
ARGS="-I \"${1:-/usr/local/MATLAB/R2017b}/extern/include\""

declare -a targets=("fmad/pool_switches")

for func in "${targets[@]}"
do
   echo "compiling $func"
   # or do whatever with individual element of the array
   nvcc --machine 64 -ptx -Xptxas=-v $ARGS ${func}.cu \
                           --output-file ${func}.ptxa64
done
