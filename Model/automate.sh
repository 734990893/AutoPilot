
## train with PID data
#python train_VPF.py --iter 0
#./automate.sh

s=0

# initial training
echo "init training iter $s --------------------------------------------------------------------"
#python train_epn.py --iter 0 | tee ./dagger_data/ep_0/train_out.log

#/media/zidong/f97c40bb-b3c0-41d4-a17f-2df375ae1d1e/zidong/future2/CARLA_0.9.6/CarlaUE4.sh
#python /media/zidong/f97c40bb-b3c0-41d4-a17f-2df375ae1d1e/zidong/future2/CARLA_0.9.6/PythonAPI/util/config.py -m Town01 --fps 20

python train_epn.py --iter $s | tee ./dagger_data/ep_$s/train_out.log

wait
sleep 2

for (( i=$(expr $s + 1); i<=8; i++ ))
do
    #echo "Welcome $i times"
    # open simulator
    echo "open simulator --------------------------------------------------------------------"
    /home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-5-g255683a7-dirty/LinuxNoEditor/CarlaUE4.sh > ./dagger_data/ep_$(expr $i - 1)/server_out.log &
    #pid=$!
    
    sleep 60
    
    python /home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-5-g255683a7-dirty/LinuxNoEditor/PythonAPI/util/config.py -m Town01 --fps 20 | tee ./dagger_data/ep_$(expr $i - 1)/server_config_out.log
    
    #echo $pid
    # recollect
    echo "recollect data --------------------------------------------------------------------"
    python new_NN_collect.py --iter $(expr $i - 1) | tee ./dagger_data/ep_$(expr $i - 1)/collect_out.log
    
    sleep 2

    # kill simulator
    pkill CarlaUE4
    sleep 5
    pkill CarlaUE4
    
    echo "training iteration $i --------------------------------------------------------------------"
    python train_epn.py --iter $i | tee ./dagger_data/ep_$i/train_out.log
    
    code=$?
    echo $code
    if (($code > 0)); then
        echo "ext fail - $code : stop"
        # kill simulator
        pkill CarlaUE4
        sleep 5
        pkill CarlaUE4
        break
    fi
    
    wait
    sleep 2
done



