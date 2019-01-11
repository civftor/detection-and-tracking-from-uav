# EXAMPLES OF COMMANDS

python tools/generate_detections.py \
    --model=../model/mars-small128.pb \
    --mot_dir=../results/rfcn/ \
    --output_dir=../results/rfcn-features/


# TO AVOID DISPLAY AND BE FASTER : REMOVE OPTION. (False doesn't work)

python deep_sort_app.py \
    --sequence_dir=/Volumes/Disque-Dur-S-C/_dsort/test/M0701 \
    --detection_file=../results/yolo/features/M0701.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True \
    --output_file=../results/yolo/M0701-track.txt


python generate_videos.py \
    --mot_dir=/Volumes/Disque-Dur-S-C/_matlab/2DMOT2015/test/ \
    --result_dir=../results/yolo-MDP-trained \
    --output_dir=../results/yolo-MDP-trained \
    --update_ms=40 \
    --convert_h264=True