#!/bin/bash

#!/bin/bash

folder=mp50
echo "FILE BATCH_SIZE MKL_FWD MKL_BWD MKL_UPD SP_FWD SP_BWD OPT_FWD OPT_BWD OPT_UPD JIT_FWD JIT_BWD JIT_UPD" >> new_time.txt

#for mm in $1
for mm in *.mm
do
echo $mm
  #for bs in 80 160 320 480 800 1600 2400 3200
  for bs in 2400
  do
    fwd_max_perf=0
    bwd_max_perf=0
    upd_max_perf=0
    jfwd_max_perf=0
    jbwd_max_perf=0
    jupd_max_perf=0
    mfwd_max_perf=0
    mbwd_max_perf=0
    mupd_max_perf=0
    sfwd_max_perf=0
    sbwd_max_perf=0
    for NB in 16 32 80 160 320
    #for NB in 16 32 80 160
    do
      for nb in 16 32 80 160 320
      do
        if [[ "$nb" -gt "$NB" ]]; then
          continue
        fi
        if [[ "$nb" == "32" && "$NB" == "80"  ]]; then
          continue
        fi
        for KB in 8 16 32 64 128
        do
          for CB in 8 16 32 64 128
          do
            echo ${NB} ${KB} ${CB} ${nb}
            make clean;make FWD_KB=${KB} BWD_CB=${CB} UPD_NB=${NB} 2>&1 1>/dev/null
            KMP_AFFINITY=compact,granularity=fine,1,28 OMP_NUM_THREADS=28 /homes//spkernel_exp/${folder}/test -i ${mm} -b ${bs} 2>&1 1>tmp
            if [ $? -eq 0 ]
            then
              fwd_perf=$(grep "OPT_FWD " tmp | awk -F " " '{print $NF}')
              fwd_perf=$(printf "%.14f" $fwd_perf)
              fwd_perf=${fwd_perf%.*}
              fwd_max_perf=$(( fwd_perf > fwd_max_perf ? fwd_perf : fwd_max_perf ))
              echo $fwd_max_perf
              bwd_perf=$(grep "OPT_BWD " tmp | awk -F " " '{print $NF}')
              bwd_perf=$(printf "%.14f" $bwd_perf)
              bwd_perf=${bwd_perf%.*}
              bwd_max_perf=$(( bwd_perf > bwd_max_perf ? bwd_perf : bwd_max_perf ))
              upd_perf=$(grep "OPT_UPD " tmp | awk -F " " '{print $NF}')
              upd_perf=$(printf "%.14f" $upd_perf)
              upd_perf=${upd_perf%.*}
              upd_max_perf=$(( upd_perf > upd_max_perf ? upd_perf : upd_max_perf ))
              upd2_perf=$(grep "OPT_UPD2 " tmp | awk -F " " '{print $NF}')
              upd2_perf=$(printf "%.14f" $upd2_perf)
              upd2_perf=${upd2_perf%.*}
              upd_max_perf=$(( upd2_perf > upd_max_perf ? upd2_perf : upd_max_perf ))
              mfwd_perf=$(grep "MKL_DENSE_FWD " tmp | awk -F " " '{print $NF}')
              mfwd_perf=$(printf "%.14f" $mfwd_perf)
              mfwd_perf=${mfwd_perf%.*}
              mfwd_max_perf=$(( mfwd_perf > mfwd_max_perf ? mfwd_perf : mfwd_max_perf ))
              mbwd_perf=$(grep "MKL_DENSE_BWD " tmp | awk -F " " '{print $NF}')
              mbwd_perf=$(printf "%.14f" $mbwd_perf)
              mbwd_perf=${mbwd_perf%.*}
              mbwd_max_perf=$(( mbwd_perf > mbwd_max_perf ? mbwd_perf : mbwd_max_perf ))
              mupd_perf=$(grep "MKL_DENSE_UPD " tmp | awk -F " " '{print $NF}')
              mupd_perf=$(printf "%.14f" $mupd_perf)
              mupd_perf=${mupd_perf%.*}
              mupd_max_perf=$(( mupd_perf > mupd_max_perf ? mupd_perf : mupd_max_perf ))
              sfwd_perf=$(grep "MKL_SPARSE_FWD " tmp | awk -F " " '{print $NF}')
              sfwd_perf=$(printf "%.14f" $sfwd_perf)
              sfwd_perf=${sfwd_perf%.*}
              sfwd_max_perf=$(( sfwd_perf > sfwd_max_perf ? sfwd_perf : sfwd_max_perf ))
              sbwd_perf=$(grep "MKL_SPARSE_BWD " tmp | awk -F " " '{print $NF}')
              sbwd_perf=$(printf "%.14f" $sbwd_perf)
              sbwd_perf=${sbwd_perf%.*}
              sbwd_max_perf=$(( sbwd_perf > sbwd_max_perf ? sbwd_perf : sbwd_max_perf ))
            else
              echo "OPT Fail"
            fi
            rm -f tmp

            #echo "JIT" ${NB} ${KB} ${CB}
            make clean
            make JIT_KB=${KB} JIT_CB=${CB} JIT_NB=${NB} JIT_nb=${nb} 2>&1 1>/dev/null
            KMP_AFFINITY=compact,granularity=fine,1,28 OMP_NUM_THREADS=28 /homes//spkernel_exp/${folder}/test -i ${mm} -b ${bs} 2>&1 1>tmp
            if [ $? -eq 0 ]
            then
              jfwd_perf=$(grep "JIT_FWD " tmp | awk -F " " '{print $NF}')
              jfwd_perf=$(printf "%.14f" $jfwd_perf)
              jfwd_perf=${jfwd_perf%.*}
              jfwd_max_perf=$(( jfwd_perf > jfwd_max_perf ? jfwd_perf : jfwd_max_perf ))
              jbwd_perf=$(grep "JIT_BWD " tmp | awk -F " " '{print $NF}')
              jbwd_perf=$(printf "%.14f" $jbwd_perf)
              jbwd_perf=${jbwd_perf%.*}
              jbwd_max_perf=$(( jbwd_perf > jbwd_max_perf ? jbwd_perf : jbwd_max_perf ))
            else
              echo "JIT Fail"
            fi
            rm -f tmp
            
            #echo "JIT_UPD" ${NB} ${KB} ${CB}
            make clean
            make JIT_UPD_KB=${KB} JIT_UPD_CB=${CB} JIT_UPD_NB=${NB} JIT_UPD_nb=${nb} 2>&1 1>/dev/null
            KMP_AFFINITY=compact,granularity=fine,1,28 OMP_NUM_THREADS=28 /homes//spkernel_exp/${folder}/test -i ${mm} -b ${bs} 2>&1 1>tmp
            if [ $? -eq 0 ]
            then
              jupd_perf=$(grep "JIT_UPD " tmp | awk -F " " '{print $NF}')
              jupd_perf=$(printf "%.14f" $jupd_perf)
              jupd_perf=${jupd_perf%.*}
              jupd_max_perf=$(( jupd_perf > jupd_max_perf ? jupd_perf : jupd_max_perf ))
            else
              echo "JIT UPD Fail"
            fi
            rm -f tmp
          done
        done
        echo $mfwd_max_perf $mbwd_max_perf $mupd_max_perf $sfwd_max_perf $sbwd_max_perf $fwd_max_perf $bwd_max_perf $upd_max_perf $jfwd_max_perf $jbwd_max_perf $jupd_max_perf
      done
    done
  echo $mm $bs $mfwd_max_perf $mbwd_max_perf $mupd_max_perf $sfwd_max_perf $sbwd_max_perf $fwd_max_perf $bwd_max_perf $upd_max_perf $jfwd_max_perf $jbwd_max_perf $jupd_max_perf >> new_time.txt
  done
done
